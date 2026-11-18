"""Bounded Sync/Async admission around the same version-fenced update pipeline."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from threading import RLock
from time import perf_counter_ns
from typing import Optional, Tuple

from torch import Tensor

from lib.graph.serving import EmbeddingService, UpdateResult


class AdmissionFull(RuntimeError):
	pass


class UpdateKind(Enum):
	FEATURES = "features"
	INSERT = "insert"
	DELETE = "delete"


@dataclass(frozen=True)
class UpdateTicket:
	sequence: int
	admitted_at: float
	future: Future[UpdateResult]
	telemetry: Optional["AdmissionTelemetry"] = None

	def result(self, timeout: float | None = None) -> UpdateResult:
		"""Wait for ready, published embeddings; storage writeback is independent."""
		return self.future.result(timeout=timeout)


@dataclass
class AdmissionTelemetry:
	"""One server-monotonic lifecycle record for an admitted update.

	The admission executor can stage a deferred update before the model
	publication starts. ``execution_ns`` therefore covers the admission-side
	service call, while ``delayed_publication_wait_ns`` covers the interval from
	staging until the publication future completes. The distinction is explicit
	so callers do not mistake deferred waiting for model execution.
	"""

	sequence: int
	admitted_at_ns: int
	execution_started_at_ns: Optional[int] = None
	staged_at_ns: Optional[int] = None
	publication_observed_at_ns: Optional[int] = None
	completion_at_ns: Optional[int] = None
	server_reply_prepared_at_ns: Optional[int] = None
	input_version: Optional[int] = None
	publication_version: Optional[int] = None
	state: str = "queued"
	failure_type: Optional[str] = None
	failure_message: Optional[str] = None
	publication_source: Optional[str] = None

	def snapshot(self, *, observed_at_ns: Optional[int] = None) -> dict:
		if observed_at_ns is not None:
			# The RPC layer supplies the server-side reply boundary. The current
			# protocol has no client clock sample, so this is never client time.
			self.server_reply_prepared_at_ns = observed_at_ns
		values = asdict(self)
		values["clock_scope"] = "server perf_counter_ns; comparable only within this process"
		values["queue_wait_ns"] = (
			self.execution_started_at_ns - self.admitted_at_ns if self.execution_started_at_ns is not None else None
		)
		if self.execution_started_at_ns is None:
			values["execution_ns"] = None
		elif self.publication_observed_at_ns is not None and self.staged_at_ns is None:
			values["execution_ns"] = self.publication_observed_at_ns - self.execution_started_at_ns
		elif self.staged_at_ns is not None:
			values["execution_ns"] = self.staged_at_ns - self.execution_started_at_ns
		else:
			values["execution_ns"] = None
		values["delayed_publication_wait_ns"] = (
			self.publication_observed_at_ns - self.staged_at_ns
			if self.publication_observed_at_ns is not None and self.staged_at_ns is not None
			else 0
			if self.staged_at_ns is None and self.publication_observed_at_ns is not None
			else None
		)
		return values


@dataclass
class AdmissionCounters:
	admitted: int = 0
	completed: int = 0
	failed: int = 0
	completed_changes: int = 0
	pending: int = 0
	pending_bytes: int = 0
	completion_seconds: float = 0.0


class UpdateAdmission:
	"""Serialize admitted mutations without blocking the Async caller on compute.

	Only submitted rows/IDs are snapshotted by default. ``borrow=True`` retains
	the submitted tensor storage with no copy; the caller gives up mutation rights
	until the ticket completes. Whole backing allocations count against the byte
	budget for borrowed views. No graph or embedding matrix is cloned.
	Accepted updates cannot be cancelled: a timeout only stops the caller's wait.
	"""

	def __init__(
		self,
		service: EmbeddingService,
		*,
		max_pending: int = 32,
		max_bytes: int = 64 * 1024**2,
		max_history: int = 4096,
		clock_ns=perf_counter_ns,
	) -> None:
		if max_pending < 1 or max_bytes < 0:
			raise ValueError("Admission limits must be positive (byte limit may be zero)")
		if type(max_history) is not int or max_history < 1:
			raise ValueError("Admission telemetry history must be positive")
		self.service = service
		self.max_pending, self.max_bytes = max_pending, max_bytes
		self.max_history = max_history
		self._clock_ns = clock_ns
		self.counters = AdmissionCounters()
		self._lock = RLock()
		self._closed = False
		self._telemetry: dict[int, AdmissionTelemetry] = {}
		self._telemetry_order: deque[int] = deque()
		self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="UPDATE-ADMISSION")

	def submit(
		self,
		kind: UpdateKind,
		first: Tensor,
		second: Tensor,
		*,
		borrow: bool = False,
		strict: bool = False,
		attributes: Optional[Tensor] = None,
	) -> UpdateTicket:
		if not isinstance(kind, UpdateKind):
			raise ValueError("Unknown update operation")
		if type(strict) is not bool:
			raise ValueError("Strict publication must be a boolean")
		if attributes is not None and (kind is not UpdateKind.INSERT or attributes.device.type != "cpu"):
			raise ValueError("Edge attributes are supported only for CPU INSERT payloads")
		values = (first, second) + ((attributes,) if attributes is not None else ())
		if first.device.type != "cpu" or second.device.type != "cpu":
			raise ValueError("Submitted storage updates must be CPU tensors")
		if borrow:
			storages = {(tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes()) for tensor in values}
			charged = sum(size for _, size in storages)
		else:
			charged = sum(tensor.numel() * tensor.element_size() for tensor in values)
		with self._lock:
			if self._closed:
				raise RuntimeError("Admission is closed")
			if self.counters.pending >= self.max_pending or self.counters.pending_bytes + charged > self.max_bytes:
				raise AdmissionFull("Update admission capacity is full")
			payload = (
				first.detach() if borrow else first.detach().clone(),
				second.detach() if borrow else second.detach().clone(),
			)
			admitted_at_ns = self._clock_ns()
			future: Future[UpdateResult] = Future()
			future.set_running_or_notify_cancel()
			attribute_rows = (
				(attributes.detach() if borrow else attributes.detach().clone()) if attributes is not None else None
			)
			sequence = self.counters.admitted + 1
			telemetry = AdmissionTelemetry(sequence, admitted_at_ns)
			self._telemetry[sequence] = telemetry
			self._telemetry_order.append(sequence)
			try:
				self._pool.submit(self._complete, kind, payload, future, strict, attribute_rows, telemetry)
			except BaseException:
				self._telemetry.pop(sequence, None)
				self._telemetry_order.pop()
				raise
			self.counters.admitted = sequence
			self.counters.pending += 1
			self.counters.pending_bytes += charged
			ticket = UpdateTicket(sequence, admitted_at_ns / 1e9, future, telemetry)
			future.add_done_callback(lambda done: self._finished(done, charged, admitted_at_ns, telemetry))
			return ticket

	def _complete(
		self,
		kind: UpdateKind,
		payload: Tuple[Tensor, Tensor],
		future: Future[UpdateResult],
		strict: bool,
		attributes: Optional[Tensor] = None,
		telemetry: Optional[AdmissionTelemetry] = None,
	) -> None:
		if telemetry is not None:
			telemetry.execution_started_at_ns = self._clock_ns()
			telemetry.state = "executing"
		try:
			if self.service.deferred is not None:
				publication = self.service.deferred.submit(
					kind.value, *payload, **({"attributes": attributes} if attributes is not None else {})
				)
				if telemetry is not None:
					telemetry.staged_at_ns = self._clock_ns()
					telemetry.state = "staged"
				publication.add_done_callback(lambda done: self._forward(done, future, telemetry))
				if strict:
					self.service.deferred.flush()
				return
			result = self._execute(kind, payload, **({"attributes": attributes} if attributes is not None else {}))
		except BaseException as error:
			self._mark_failure(telemetry, error)
			if not future.done():
				future.set_exception(error)
		else:
			self._finish_success(future, result, telemetry, "service_completion")

	def _forward(
		self, publication: Future[UpdateResult], future: Future[UpdateResult], telemetry: Optional[AdmissionTelemetry]
	) -> None:
		try:
			result = publication.result()
		except BaseException as error:
			self._mark_failure(telemetry, error)
			if not future.done():
				future.set_exception(error)
		else:
			self._finish_success(future, result, telemetry, "deferred_future_callback")

	def _finish_success(
		self, future: Future[UpdateResult], result: UpdateResult, telemetry: Optional[AdmissionTelemetry], source: str
	) -> None:
		if telemetry is not None:
			telemetry.publication_observed_at_ns = self._clock_ns()
			telemetry.input_version = result.input_version
			telemetry.publication_version = result.version
			telemetry.publication_source = source
			telemetry.state = "published"
		if not future.done():
			future.set_result(result)

	def _mark_failure(self, telemetry: Optional[AdmissionTelemetry], error: BaseException) -> None:
		if telemetry is None:
			return
		telemetry.completion_at_ns = self._clock_ns()
		telemetry.state = "failed"
		telemetry.failure_type = type(error).__name__
		telemetry.failure_message = str(error)

	def flush(self) -> None:
		"""Queue an exact publication barrier behind all previously admitted updates."""
		with self._lock:
			if self._closed:
				raise RuntimeError("Admission is closed")
			if self.service.deferred is not None:
				self._pool.submit(self.service.deferred.flush)

	def _execute(
		self, kind: UpdateKind, payload: Tuple[Tensor, Tensor], attributes: Optional[Tensor] = None
	) -> UpdateResult:
		if kind is UpdateKind.FEATURES:
			return self.service.change_features(*payload)
		if kind is UpdateKind.INSERT:
			return self.service.insert_edges(*payload, **({"attributes": attributes} if attributes is not None else {}))
		return self.service.delete_edges(*payload)

	def _finished(
		self, future: Future[UpdateResult], charged: int, admitted_at_ns: int, telemetry: AdmissionTelemetry
	) -> None:
		with self._lock:
			telemetry.completion_at_ns = telemetry.completion_at_ns or self._clock_ns()
			self.counters.pending -= 1
			self.counters.pending_bytes -= charged
			if future.cancelled() or future.exception() is not None:
				self.counters.failed += 1
			else:
				self.counters.completed += 1
				self.counters.completed_changes += future.result().input_changes
				self.counters.completion_seconds += (telemetry.completion_at_ns - admitted_at_ns) / 1e9
			self._prune_telemetry()

	def _prune_telemetry(self) -> None:
		while len(self._telemetry_order) > self.max_history:
			sequence = self._telemetry_order[0]
			telemetry = self._telemetry.get(sequence)
			if telemetry is not None and telemetry.completion_at_ns is None:
				return
			self._telemetry_order.popleft()
			self._telemetry.pop(sequence, None)

	def profile(self, sequence: int, *, observed_at_ns: Optional[int] = None) -> Optional[dict]:
		"""Return a bounded lifecycle snapshot and optionally mark caller observation."""
		with self._lock:
			telemetry = self._telemetry.get(sequence)
			return telemetry.snapshot(observed_at_ns=observed_at_ns) if telemetry is not None else None

	def profile_state(self) -> dict:
		"""Return admission state for a profiler window without changing scheduling."""
		with self._lock:
			return {
				"admitted": self.counters.admitted,
				"completed": self.counters.completed,
				"failed": self.counters.failed,
				"pending": self.counters.pending,
				"pending_bytes": self.counters.pending_bytes,
				"max_pending": self.max_pending,
				"max_bytes": self.max_bytes,
			}

	def sync(self, kind: UpdateKind, first: Tensor, second: Tensor, *, borrow: bool = False) -> UpdateResult:
		return self.submit(kind, first, second, borrow=borrow, strict=True).result()

	def close(self) -> None:
		with self._lock:
			self._closed = True
		self._pool.shutdown(wait=True)
		if self.service.deferred is not None:
			self.service.deferred.flush()
