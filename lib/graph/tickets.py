"""Bounded, session-scoped completion tickets for a frontend's Sync/Async calls."""

import hashlib
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable, Dict, NoReturn, Optional, Type
from uuid import uuid4

import torch
from torch import Tensor

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.serving import UpdateResult


class SessionMismatch(ValueError):
	pass


@dataclass(frozen=True)
class _Failure:
	error_type: Type[BaseException]
	message: str

	def raise_fresh(self) -> NoReturn:
		try:
			error = self.error_type(self.message)
		except Exception:
			error = RuntimeError(self.message)
		raise error from None


@dataclass(frozen=True)
class _Outcome:
	result: Optional[UpdateResult] = None
	failure: Optional[_Failure] = None


@dataclass
class CompletionTicket:
	session: str
	number: int
	request_id: str
	future: Future[_Outcome]
	signature: bytes
	completed_at: Optional[float] = None

	def result(self, timeout: Optional[float] = None) -> UpdateResult:
		outcome = self.future.result(timeout=timeout)
		if outcome.failure is not None:
			outcome.failure.raise_fresh()
		assert outcome.result is not None
		return outcome.result


def _signature(kind: UpdateKind, first: Tensor, second: Tensor, attributes: Optional[Tensor] = None) -> bytes:
	digest = hashlib.sha256(kind.value.encode("ascii"))
	for value in (first, second) + ((attributes,) if attributes is not None else ()):
		if (
			value.device.type != "cpu"
			or not value.is_contiguous()
			or value.dtype
			not in (
				torch.int64,
				torch.float32,
				torch.bool,
			)
		):
			raise ValueError("Ticket payloads require contiguous CPU int64, float32, or bool tensors")
		digest.update(repr((str(value.dtype), tuple(value.shape))).encode("ascii"))
		if value.numel():
			digest.update(memoryview(value.detach().numpy()).cast("B"))
	return digest.digest()


class TicketRegistry:
	"""Retain completion metadata, with bounded deduplication in one live session.

	Identical request IDs share admission while retained. Changed payloads under
	the same ID fail. Release explicitly forgets completed IDs; completed records
	also expire after retention. Pending updates never expire and cannot cancel.
	Durable deduplication across master restarts is outside this contract.
	"""

	def __init__(
		self,
		admission: UpdateAdmission,
		*,
		capacity: int = 1024,
		retention: float = 300,
		clock: Callable[[], float] = monotonic,
	) -> None:
		if capacity < 1 or retention < 0:
			raise ValueError("Invalid ticket capacity or retention")
		self.admission, self.capacity, self.retention = admission, capacity, retention
		self.session = uuid4().hex
		self._clock = clock
		self._lock = RLock()
		self._entries: Dict[int, CompletionTicket] = {}
		self._requests: Dict[str, int] = {}

	def _drop(self, entry: CompletionTicket) -> None:
		del self._entries[entry.number]
		del self._requests[entry.request_id]

	def _prune(self) -> None:
		now = self._clock()
		for entry in tuple(self._entries.values()):
			if entry.completed_at is not None and now - entry.completed_at >= self.retention:
				self._drop(entry)

	def _check_session(self, session: str) -> None:
		if session != self.session:
			raise SessionMismatch("Ticket belongs to another server session")

	def submit(
		self,
		kind: UpdateKind,
		first: Tensor,
		second: Tensor,
		*,
		request_id: str,
		session: str = "",
		strict: bool = False,
		attributes: Optional[Tensor] = None,
	) -> CompletionTicket:
		if session:
			self._check_session(session)
		if not request_id or len(request_id) > 128:
			raise ValueError("Updates require a request ID of 1 to 128 characters")
		if (
			sum(
				value.untyped_storage().nbytes()
				for value in (first, second) + ((attributes,) if attributes is not None else ())
			)
			> self.admission.max_bytes
		):
			raise AdmissionFull("Update payload exceeds the admission byte limit")
		signature = _signature(kind, first, second, attributes)
		with self._lock:
			self._prune()
			if request_id in self._requests:
				entry = self._entries[self._requests[request_id]]
				if entry.signature != signature:
					raise ValueError("Request ID was already used for a different update")
				if strict and not entry.future.done():
					self.admission.flush()
				return entry
			if len(self._entries) >= self.capacity:
				raise AdmissionFull("Completion ticket capacity is full; release completed tickets")
			# Frontend-decoded tensors borrow immutable bytes. Their lifetime is
			# retained by admission until execution completes, without another copy.
			admitted = self.admission.submit(
				kind,
				first,
				second,
				borrow=True,
				strict=strict,
				**({"attributes": attributes} if attributes is not None else {}),
			)
			future: Future[_Outcome] = Future()
			future.set_running_or_notify_cancel()
			entry = CompletionTicket(self.session, admitted.sequence, request_id, future, signature)
			self._entries[entry.number] = entry
			self._requests[request_id] = entry.number
			admitted.future.add_done_callback(lambda done: self._finished(entry, done))
			return entry

	def _finished(self, entry: CompletionTicket, original: Future[UpdateResult]) -> None:
		error = original.exception()
		if error is None:
			outcome = _Outcome(result=original.result())
		else:
			logging.error("Admitted frontend update failed", exc_info=(type(error), error, error.__traceback__))
			error_type = type(error) if isinstance(error, Exception) else RuntimeError
			outcome = _Outcome(failure=_Failure(error_type, str(error)))
			# Failure frames may retain the submitted feature bytes. Retain only
			# type/message; each later wait raises a fresh, short-lived exception.
			error.__traceback__ = error.__context__ = error.__cause__ = None
		with self._lock:
			entry.completed_at = self._clock()
			entry.future.set_result(outcome)

	def get(self, session: str, number: int, request_id: str) -> CompletionTicket:
		self._check_session(session)
		with self._lock:
			self._prune()
			entry = self._entries[number]
			if entry.request_id != request_id:
				raise ValueError("Ticket identity does not match its request ID")
			return entry

	def release(self, session: str, number: int, request_id: str) -> None:
		with self._lock:
			entry = self.get(session, number, request_id)
			if not entry.future.done():
				raise RuntimeError("An admitted update is still running")
			self._drop(entry)
