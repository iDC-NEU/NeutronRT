"""Merge derived work across layers under one graph-version output budget."""

import logging
import math
from concurrent.futures import Future
from dataclasses import dataclass, replace
from threading import Timer
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from lib.graph.affected import AffectedLayer
from lib.graph.policies import DeferredPolicyConfig
from lib.graph.serving import UpdateResult, _PendingUpdate

if TYPE_CHECKING:
	from lib.graph.serving import EmbeddingService


@dataclass(frozen=True)
class _Entry:
	input_version: int
	changes: int
	start: float
	affected: Tuple[int, ...]
	future: Future[UpdateResult]


class DeferredUpdates:
	"""Stage authoritative updates; publish only exact, merged model outputs.

	All model inputs remain fresh. An output is either fresh or a previously
	published result with a bounded age, so layer budgets never compound. Strict
	reads compute their exact dependency closure. Async tickets remain pending
	until the complete merged affected set is published. Idle flush ensures liveness.
	Only node IDs, counters, and futures are retained between updates.
	"""

	def __init__(
		self,
		service: "EmbeddingService",
		*,
		max_versions: int = 2,
		idle_seconds: float = 0.005,
		policy: Optional[DeferredPolicyConfig] = None,
		clock: Callable[[], float] = monotonic,
		timer_factory: Optional[Callable[..., Timer]] = None,
	) -> None:
		if type(max_versions) is not int or max_versions < 0 or not math.isfinite(idle_seconds) or idle_seconds <= 0:
			raise ValueError("Invalid shared version budget or idle flush interval")
		if service.deferred is not None:
			raise ValueError("The service already has deferred update scheduling")
		service.catalog.enable_staleness()
		self.service, self.max_versions, self.idle_seconds = service, max_versions, idle_seconds
		self.policy = policy or DeferredPolicyConfig()
		self.window_seconds = idle_seconds if self.policy.window_seconds is None else self.policy.window_seconds
		self._clock = clock
		self._timer_factory = Timer if timer_factory is None else timer_factory
		self._deadline: Optional[float] = None
		self._entries: List[_Entry] = []
		self._dirty: Dict[int, Tensor] = {}
		self._active: Optional[_PendingUpdate] = None
		self._timer: Optional[Timer] = None
		self._timer_sequence = 0
		self._closed = False
		service.deferred = self

	@property
	def pending(self) -> int:
		with self.service._fence:
			return len(self._entries)

	def _cancel_timer(self) -> None:
		self._timer_sequence += 1
		if self._timer is not None:
			self._timer.cancel()
			self._timer = None

	def _arm_timer(self) -> None:
		if self.policy.mode in ("fixed_time", "hotspot"):
			if self._deadline is None:
				self._deadline = self._clock() + self.window_seconds
			if self._timer is not None:
				return
			self._cancel_timer()
			seconds = max(0.0, self._deadline - self._clock())
		else:
			self._cancel_timer()
			self._deadline = None
			seconds = self.idle_seconds
		self._timer = self._timer_factory(seconds, self._idle, args=(self._timer_sequence,))
		self._timer.daemon = True
		self._timer.start()

	def _hotspot(self, kind: str, first: Tensor, second: Tensor) -> bool:
		if self.policy.mode != "hotspot":
			return False
		nodes = first if kind == "features" else torch.cat((first, second))
		if not nodes.numel():
			return False
		threshold = self.policy.hotspot_degree_threshold
		assert threshold is not None
		degree = torch.maximum(self.service.graph.in_degrees(nodes), self.service.graph.out_degrees(nodes))
		return bool((degree >= threshold).any())

	def _idle(self, sequence: int) -> None:
		with self.service._fence:
			if self._closed or sequence != self._timer_sequence or self._active is not None:
				return
			if (
				self._entries
				and self.policy.mode in ("fixed_time", "hotspot")
				and self._deadline is not None
				and self._clock() < self._deadline
			):
				# Timer callbacks may run early; preserve the first-entry deadline.
				self._timer = None
				self._arm_timer()
				return
			try:
				self.flush()
			except Exception:
				logging.exception("Deferred publication failed; recover and resume the applied version")

	def _merge(self, layers: Tuple[AffectedLayer, ...]) -> None:
		for item in layers:
			previous = self._dirty.get(item.layer)
			self._dirty[item.layer] = item.targets if previous is None else torch.cat([previous, item.targets]).unique()

	def submit(
		self, kind: str, first: Tensor, second: Tensor, *, attributes: Optional[Tensor] = None
	) -> Future[UpdateResult]:
		service = self.service
		with service._fence:
			if self._closed:
				raise RuntimeError("Deferred update scheduler is closed")
			if (
				self._entries
				and self.policy.mode in ("fixed_time", "hotspot")
				and self._deadline is not None
				and self._clock() >= self._deadline
			):
				# The waiting cap covers staged work only. Execute the due batch before
				# admitting another mutation, so continuous arrivals cannot extend it.
				self.flush()
			hot = self._hotspot(kind, first, second)
			if self.policy.mode == "hotspot" and self._entries and not hot:
				self.flush()
			service._require_update_slot(allow_deferred=True)
			if kind == "features":
				mutation = service._mutate_features(first, second)
			elif kind in ("insert", "delete"):
				mutation = service._mutate_edges(
					first,
					second,
					delete=kind == "delete",
					**({"attributes": attributes} if attributes is not None else {}),
				)
			else:
				raise ValueError("Unknown deferred update operation")
			version = service.catalog.version + 1
			future: Future[UpdateResult] = Future()
			future.set_running_or_notify_cancel()
			try:
				layers = tuple(
					service.affected.analyze(
						changed_features=mutation.changed_features,
						edge_targets=mutation.edge_targets,
						degree_sources=mutation.edge_sources if service.source_degree_sensitive else None,
					)
				)
			except BaseException as error:
				# The authoritative mutation is already applied. Conservatively dirty
				# all rows, preserve a resumable record, and never replay its input.
				all_nodes = torch.arange(service.graph.num_nodes())
				layers = tuple(AffectedLayer(layer, all_nodes) for layer in range(1, len(service.widths)))
				self._record(version, mutation.changes, mutation.start, layers, future)
				self._begin_flush()
				self._fail(error)
				raise
			self._record(version, mutation.changes, mutation.start, layers, future)
			# Every policy retains the same version-window safety bound.  The other
			# policies only change when a safe pending batch is flushed.
			if version - self._entries[0].input_version + 1 > self.max_versions or (
				self.policy.mode == "hotspot"
				and self.policy.hotspot_max_changes is not None
				and (not hot or sum(entry.changes for entry in self._entries) >= self.policy.hotspot_max_changes)
			):
				self.flush()
			else:
				self._arm_timer()
			return future

	def _record(self, version, changes, start, layers, future) -> None:
		self.service.catalog.begin_version(version, {item.layer: item.targets for item in layers})
		self._entries.append(_Entry(version, changes, start, tuple(item.targets.numel() for item in layers), future))
		self._merge(layers)
		self.service.counters.staging_seconds += perf_counter() - start

	def _begin_flush(self) -> _PendingUpdate:
		self._cancel_timer()
		self._deadline = None
		service = self.service
		self._active = _PendingUpdate(
			service.catalog.version,
			sum(entry.changes for entry in self._entries),
			perf_counter(),
			service.counters.tasks,
			tuple(AffectedLayer(layer, targets) for layer, targets in sorted(self._dirty.items())),
		)
		service._pending_update = self._active
		return self._active

	def _fail(self, error: BaseException) -> None:
		for entry in self._entries:
			if not entry.future.done():
				# Do not retain failed execution frames and their tensor arguments.
				try:
					failure = type(error)(str(error))
				except Exception:
					failure = RuntimeError(str(error))
				entry.future.set_exception(failure)

	def flush(self) -> Optional[UpdateResult]:
		with self.service._fence:
			if not self._entries:
				return None
			if self._active is not None:
				raise RuntimeError("Deferred publication is incomplete; recover and resume it")
			pending = self._begin_flush()
			try:
				return self.service._execute_merged(pending)
			except BaseException as error:
				self._fail(error)
				raise

	def completed(self, pending: _PendingUpdate, result: UpdateResult) -> None:
		"""Called after ordinary or resumed publication has revalidated every layer."""
		if pending is not self._active:
			return
		entries, self._entries = self._entries, []
		self._dirty = {}
		self._active = None
		service = self.service
		service.counters.updates += len(entries) - 1
		service.counters.merged_batches += 1
		service.counters.coalesced_affected_rows += sum(sum(entry.affected) for entry in entries) - sum(result.affected)
		for index, entry in enumerate(entries):
			if not entry.future.done():
				entry.future.set_result(
					replace(
						result,
						input_version=entry.input_version,
						input_changes=entry.changes,
						affected=entry.affected,
						tasks=result.tasks if index == 0 else 0,
						seconds=perf_counter() - entry.start,
					)
				)

	def profile(self) -> dict:
		return {
			**self.policy.as_dict(),
			"max_versions": self.max_versions,
			"idle_seconds": self.idle_seconds,
			"waiting_scope": "staged_entries_before_publication",
			"deadline_active": self._deadline is not None,
			"deadline_remaining_seconds": None if self._deadline is None else max(0.0, self._deadline - self._clock()),
			"pending": len(self._entries),
			"pending_changes": sum(entry.changes for entry in self._entries),
		}

	def close(self) -> None:
		with self.service._fence:
			self._closed = True
			self._cancel_timer()
			self.flush()
