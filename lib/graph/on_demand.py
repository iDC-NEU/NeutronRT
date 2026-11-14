"""On-demand reads with bounded cooperative preemption at ready chunk boundaries."""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import perf_counter
from typing import Callable, Deque

import torch
from torch import Tensor

from lib.graph.admission import AdmissionFull
from lib.graph.ownership import Owner
from lib.graph.serving import EmbeddingService


@dataclass(frozen=True)
class QueryReply:
	version: int
	value: object
	max_age: int = 0


@dataclass
class QueryCounters:
	admitted: int = 0
	completed: int = 0
	failed: int = 0
	pending: int = 0
	pending_bytes: int = 0
	recomputed_rows: int = 0
	completion_seconds: float = 0.0


@dataclass(frozen=True)
class _Query:
	nodes: Tensor
	owner: Owner
	layer: int
	max_staleness: int
	consume: Callable[[Tensor], object]
	future: Future[QueryReply]
	charged: int
	start: float


class OnDemand:
	"""Read the current fenced version before the next normal compute chunk.

	The consumer runs while input/output leases remain held. It must serialize or
	copy any values it retains past return. Admission snapshots only requested IDs;
	borrowed IDs must remain immutable until completion. A request never interrupts
	a kernel or a transfer, and one query is serviced per chunk boundary so updates
	keep making progress. Idle reads use the same graph fence and computation path.
	"""

	def __init__(self, service: EmbeddingService, *, max_pending: int = 8, max_bytes: int = 64 * 1024**2) -> None:
		if max_pending < 1 or max_bytes < 0:
			raise ValueError("Invalid on-demand admission limits")
		if service.preemption_enabled and service.checkpoint is not None:
			raise ValueError("The service already has a compute checkpoint handler")
		self.service = service
		self.max_pending, self.max_bytes = max_pending, max_bytes
		self.counters = QueryCounters()
		self._lock = RLock()
		self._wake = Event()
		self._queue: Deque[_Query] = deque()
		self._closed = False
		self._running = False
		if service.preemption_enabled:
			service.checkpoint = self._checkpoint
		self._thread = Thread(target=self._idle, name="ON-DEMAND", daemon=True)
		self._thread.start()

	def submit(
		self,
		nodes: Tensor,
		owner: Owner,
		consume: Callable[[Tensor], object],
		*,
		layer: int | None = None,
		borrow: bool = False,
		max_staleness: int = 0,
	) -> Future[QueryReply]:
		layer_ = len(self.service.widths) - 1 if layer is None else layer
		self.service._check_query_budget(max_staleness)
		# Dimensions and graph size are immutable; validation must not wait for an
		# update's graph fence or the Async request would become synchronous.
		if not 0 <= layer_ < len(self.service.widths):
			raise ValueError("Query layer out of range")
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Query IDs must be one-dimensional CPU int64 metadata")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.service.catalog.nodes):
			raise ValueError("Query node ID out of range")
		charged = nodes.untyped_storage().nbytes() if borrow else nodes.numel() * nodes.element_size()
		charged += nodes.numel() * self.service.widths[layer_] * self.service.features.element_size()
		with self._lock:
			if self._closed:
				raise RuntimeError("On-demand scheduler is closed")
			if self.counters.pending >= self.max_pending or self.counters.pending_bytes + charged > self.max_bytes:
				raise AdmissionFull("On-demand capacity is full")
			payload = nodes.detach() if borrow else nodes.detach().clone()
			future: Future[QueryReply] = Future()
			future.set_running_or_notify_cancel()
			self._queue.append(_Query(payload, owner, layer_, max_staleness, consume, future, charged, perf_counter()))
			self.counters.admitted += 1
			self.counters.pending += 1
			self.counters.pending_bytes += charged
			self._wake.set()
			return future

	def _checkpoint(self) -> None:
		# The graph fence excludes the idle reader while a coordinator and its
		# compute thread run. Recursive recomputation reaches checkpoints too.
		if not self._running:
			self._drain_one()

	def _drain_one(self) -> None:
		with self._lock:
			if not self._queue:
				self._wake.clear()
				return
			request = self._queue.popleft()
		self._running = True
		before = self.service.counters.rows
		try:
			with self.service._query_unfenced(
				request.nodes, request.owner, request.layer, request.max_staleness
			) as values:
				version = self.service.catalog.version
				ages = self.service.catalog.ages(request.layer, request.nodes, version)
				reply = QueryReply(version, request.consume(values), int(ages.max()) if ages.numel() else 0)
		except BaseException as error:
			with self._lock:
				self.counters.failed += 1
			request.future.set_exception(error)
		else:
			with self._lock:
				self.counters.completed += 1
			request.future.set_result(reply)
		finally:
			with self._lock:
				self.counters.pending -= 1
				self.counters.pending_bytes -= request.charged
				self.counters.recomputed_rows += self.service.counters.rows - before
				self.counters.completion_seconds += perf_counter() - request.start
				self._running = False
				if not self._queue:
					self._wake.clear()

	def _idle(self) -> None:
		while True:
			self._wake.wait()
			with self.service._fence:
				self._drain_one()
			with self._lock:
				if self._closed and not self._queue:
					return

	def close(self) -> None:
		with self._lock:
			self._closed = True
			self._wake.set()
		self._thread.join()
		if self.service.checkpoint == self._checkpoint:
			self.service.checkpoint = None
