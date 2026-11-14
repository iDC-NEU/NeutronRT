"""In-process endpoint backend for the versioned embedding data-plane protocol.

Each endpoint owns its buffers on its configured CPU/GPU. Cross-process transport
can implement the same publication/copy contract; this backend makes no network
performance claims.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, RLock
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from lib.graph.ownership import ComputeAttempt, CopyAttempt, EmbeddingDirectory, Location, MissingState, Owner, StateKey
from lib.graph.resident import Readiness, ResidentTensors, TensorLease

Transfer = Callable[[Tensor, torch.device], Tuple[Tensor, Optional[Readiness]]]


class RejectedPublication(RuntimeError):
	pass


@torch.inference_mode()
def copy_tensor(tensor: Tensor, device: torch.device) -> Tuple[Tensor, Optional[Readiness]]:
	"""Copy once into independent storage and expose completion separately."""
	if not tensor.is_cuda and device.type == "cpu":
		return tensor.clone(), None
	stream_device = device if device.type == "cuda" else tensor.device
	with torch.cuda.device(stream_device):
		if device.type == "cpu":
			result = torch.empty_like(tensor, device="cpu", pin_memory=True)
			result.copy_(tensor, non_blocking=True)
		else:
			result = tensor.to(device, non_blocking=True, copy=True)
		event = torch.cuda.Event()
		event.record(torch.cuda.current_stream(stream_device))
	return result, event


@dataclass
class LocalEndpoint:
	owner: Owner
	device: torch.device
	store: ResidentTensors


class LocalEmbeddingPlane:
	def __init__(self, *, max_pending: int = 4, transfer: Transfer = copy_tensor) -> None:
		if max_pending < 1:
			raise ValueError("Need a positive transfer limit")
		self.directory = EmbeddingDirectory()
		self._lock = RLock()
		self._nodes: Dict[Owner, LocalEndpoint] = {}
		self._inflight: Dict[Tuple[StateKey, Owner], Future[Location]] = {}
		self._slots = BoundedSemaphore(max_pending)
		self._pool = ThreadPoolExecutor(max_workers=max_pending, thread_name_prefix="WRITEBACK")
		self._transfer = transfer
		self._closed = False
		self.failed_transfers = 0

	def add_node(self, rank: int, device: str, max_bytes: int) -> LocalEndpoint:
		device_ = torch.device(device)
		if device_.type not in ("cpu", "cuda"):
			raise ValueError("Endpoint must use CPU or CUDA")
		if device_.type == "cuda" and device_.index is None:
			device_ = torch.device("cuda", torch.cuda.current_device())
		store = ResidentTensors(max_bytes)
		with self._lock:
			if self._closed:
				raise RuntimeError("Data plane is closed")
			owner = self.directory.register_owner(rank)
			node = LocalEndpoint(owner, device_, store)
			self._nodes[owner] = node
			return node

	def _node(self, owner: Owner) -> LocalEndpoint:
		with self._lock:
			try:
				return self._nodes[owner]
			except KeyError as error:
				raise MissingState(owner) from error

	def empty(self, owner: Owner, shape: Tuple[int, ...], *, dtype: torch.dtype) -> Tensor:
		return torch.empty(shape, device=self._node(owner).device, dtype=dtype)

	def to_owner(self, tensor: Tensor, owner: Owner) -> Tensor:
		device = self._node(owner).device
		if tensor.device == device:
			return tensor
		value, readiness = self._transfer(tensor, device)
		if value.device != device:
			raise ValueError("Transfer returned the wrong endpoint device")
		if readiness is not None:
			readiness.synchronize()
		return value

	def publish(self, attempt: ComputeAttempt, tensor: Tensor) -> Location:
		node = self._node(attempt.owner)
		if tensor.device != node.device:
			raise ValueError("Computed tensor is on the wrong endpoint device")
		handle = node.store.put(tensor)
		accepted = False
		try:
			with node.store.acquire(handle, wait=True):
				if not self.directory.publish(attempt, handle):
					raise RejectedPublication("Compute attempt is no longer current")
				node.store.mark_published(handle)
				accepted = True
			return Location(node.owner, handle)
		finally:
			if not accepted:
				node.store.abandon(handle)

	def acquire(self, key: StateKey, *, owner: Optional[Owner] = None) -> TensorLease:
		location = self.directory.locate(key, preferred=owner)
		if owner is not None and location.owner != owner:
			location = self._fetch(key, owner)
		return self._node(location.owner).store.acquire(location.handle, wait=True)

	def _fetch(self, key: StateKey, target: Owner) -> Location:
		"""Required inputs use their caller's slot, independent of deferred writeback.

		Join an existing transfer of the same state, but never queue unrelated
		foreground reads behind the bounded background executor.
		"""
		pair = key, target
		with self._lock:
			if self._closed:
				raise RuntimeError("Data plane is closed")
			self._node(target)
			location = self.directory.locate(key, preferred=target)
			if location.owner == target:
				return location
			future = self._inflight.get(pair)
			leader = future is None
			if leader:
				attempt = self.directory.begin_copy(key, target)
				future = Future()
				self._inflight[pair] = future
		assert future is not None
		if not leader:
			return future.result()
		try:
			location = self._copy(attempt)
			future.set_result(location)
			return location
		except BaseException as error:
			future.set_exception(error)
			with self._lock:
				self.failed_transfers += 1
			raise
		finally:
			with self._lock:
				if self._inflight.get(pair) is future:
					del self._inflight[pair]

	def _copy(self, attempt: CopyAttempt) -> Location:
		source = self._node(attempt.source.owner)
		target = self._node(attempt.target)
		handle = None
		accepted = False
		try:
			with source.store.acquire(attempt.source.handle, wait=True) as tensor:
				value, readiness = self._transfer(tensor, target.device)
				if value.device != target.device:
					raise ValueError("Transfer returned the wrong endpoint device")
				handle = target.store.put(value, readiness=readiness)
				with target.store.acquire(handle, wait=True):
					if not self.directory.complete_copy(attempt, handle):
						raise RejectedPublication("Transfer attempt is no longer current")
					target.store.mark_published(handle)
					accepted = True
					target.store.mark_backed_up(handle)
					source.store.mark_backed_up(attempt.source.handle)
			return Location(target.owner, handle)
		finally:
			self.directory.abort_copy(attempt)
			if handle is not None and not accepted:
				target.store.abandon(handle)

	def writeback(self, key: StateKey, target: Owner) -> Future[Location]:
		with self._lock:
			if self._closed:
				raise RuntimeError("Data plane is closed")
			self._node(target)
			pair = key, target
			if pair in self._inflight:
				return self._inflight[pair]
			location = self.directory.locate(key, preferred=target)
			if location.owner == target:
				ready: Future[Location] = Future()
				ready.set_result(location)
				return ready
			if not self._slots.acquire(blocking=False):
				raise MemoryError("Deferred transfer queue is full")
			attempt = None
			try:
				attempt = self.directory.begin_copy(key, target)
				future = self._pool.submit(self._copy, attempt)
			except BaseException:
				if attempt is not None:
					self.directory.abort_copy(attempt)
				self._slots.release()
				raise
			self._inflight[pair] = future
			issued = attempt
			future.add_done_callback(lambda done: self._finished(pair, issued, done))
			return future

	def _finished(self, pair: Tuple[StateKey, Owner], attempt: CopyAttempt, future: Future[Location]) -> None:
		with self._lock:
			self._inflight.pop(pair, None)
			self.directory.abort_copy(attempt)
			self._slots.release()
			if not future.cancelled() and future.exception() is not None:
				self.failed_transfers += 1
				logging.warning("Deferred embedding transfer failed: %s", future.exception())

	def lose_owner(self, owner: Owner) -> Tuple[StateKey, ...]:
		with self._lock:
			missing = self.directory.lose_owner(owner)
			node = self._nodes.pop(owner, None)
			if node is not None:
				node.store.retire_all()
			return missing

	def evict(self, key: StateKey, owner: Owner) -> None:
		location = self.directory.locate(key, preferred=owner)
		if location.owner != owner:
			raise MissingState(key)
		node = self._node(owner)
		node.store.evict(location.handle, authorize=lambda: self.directory.remove_replica(key, location))

	def retire(self, key: StateKey) -> None:
		"""Called only after no future task/query requires this state key."""
		with self._lock:
			for location in self.directory.forget(key):
				node = self._nodes.get(location.owner)
				if node is not None:
					node.store.retire(location.handle)

	def close(self) -> None:
		with self._lock:
			self._closed = True
		self._pool.shutdown(wait=True)
