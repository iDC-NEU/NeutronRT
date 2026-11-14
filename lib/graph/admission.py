"""Bounded Sync/Async admission around the same version-fenced update pipeline."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import perf_counter
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

	def result(self, timeout: float | None = None) -> UpdateResult:
		"""Wait for ready, published embeddings; storage writeback is independent."""
		return self.future.result(timeout=timeout)


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

	def __init__(self, service: EmbeddingService, *, max_pending: int = 32, max_bytes: int = 64 * 1024**2) -> None:
		if max_pending < 1 or max_bytes < 0:
			raise ValueError("Admission limits must be positive (byte limit may be zero)")
		self.service = service
		self.max_pending, self.max_bytes = max_pending, max_bytes
		self.counters = AdmissionCounters()
		self._lock = RLock()
		self._closed = False
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
			admitted_at = perf_counter()
			future: Future[UpdateResult] = Future()
			future.set_running_or_notify_cancel()
			attribute_rows = (
				(attributes.detach() if borrow else attributes.detach().clone()) if attributes is not None else None
			)
			self._pool.submit(self._complete, kind, payload, future, strict, attribute_rows)
			self.counters.admitted += 1
			self.counters.pending += 1
			self.counters.pending_bytes += charged
			ticket = UpdateTicket(self.counters.admitted, admitted_at, future)
			future.add_done_callback(lambda done: self._finished(done, charged, admitted_at))
			return ticket

	def _complete(
		self,
		kind: UpdateKind,
		payload: Tuple[Tensor, Tensor],
		future: Future[UpdateResult],
		strict: bool,
		attributes: Optional[Tensor] = None,
	) -> None:
		try:
			if self.service.deferred is not None:
				publication = self.service.deferred.submit(
					kind.value, *payload, **({"attributes": attributes} if attributes is not None else {})
				)
				publication.add_done_callback(lambda done: self._forward(done, future))
				if strict:
					self.service.deferred.flush()
				return
			result = self._execute(kind, payload, **({"attributes": attributes} if attributes is not None else {}))
		except BaseException as error:
			if not future.done():
				future.set_exception(error)
		else:
			future.set_result(result)

	@staticmethod
	def _forward(publication: Future[UpdateResult], future: Future[UpdateResult]) -> None:
		try:
			result = publication.result()
		except BaseException as error:
			future.set_exception(error)
		else:
			future.set_result(result)

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

	def _finished(self, future: Future[UpdateResult], charged: int, admitted_at: float) -> None:
		with self._lock:
			self.counters.pending -= 1
			self.counters.pending_bytes -= charged
			if future.cancelled() or future.exception() is not None:
				self.counters.failed += 1
			else:
				self.counters.completed += 1
				self.counters.completed_changes += future.result().input_changes
				self.counters.completion_seconds += perf_counter() - admitted_at

	def sync(self, kind: UpdateKind, first: Tensor, second: Tensor, *, borrow: bool = False) -> UpdateResult:
		return self.submit(kind, first, second, borrow=borrow, strict=True).result()

	def close(self) -> None:
		with self._lock:
			self._closed = True
		self._pool.shutdown(wait=True)
		if self.service.deferred is not None:
			self.service.deferred.flush()
