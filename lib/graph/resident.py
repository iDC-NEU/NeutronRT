"""Bounded, zero-copy ownership of immutable compute results."""

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple, cast

import torch
from torch import Tensor


class Readiness(Protocol):
	def query(self) -> bool: ...

	def synchronize(self) -> None: ...


class BufferNotReady(RuntimeError):
	pass


class BufferInUse(RuntimeError):
	pass


class UnflushedBuffer(RuntimeError):
	pass


@dataclass
class _Buffer:
	tensor: Tensor
	storage: Tuple[torch.device, int]
	readiness: Optional[Readiness]
	readers: int = 0
	backed_up: bool = False
	published: bool = False
	retired: bool = False


class TensorLease:
	"""Keep a published tensor pinned until its consumer has finished using it."""

	def __init__(self, store: "ResidentTensors", handle: int, tensor: Tensor) -> None:
		self._tensor: Optional[Tensor] = tensor
		self._store = store
		self._handle = handle
		self._closed = False

	@property
	def tensor(self) -> Tensor:
		if self._closed:
			raise RuntimeError("Tensor lease is closed")
		return cast(Tensor, self._tensor)

	def close(self) -> None:
		with self._store._lock:
			if not self._closed:
				self._store._release(self._handle)
				self._closed = True
				self._tensor = None

	def __enter__(self) -> Tensor:
		if self._closed:
			raise RuntimeError("Tensor lease is closed")
		return self.tensor

	def __exit__(self, *_exc) -> None:
		self.close()


class ResidentTensors:
	"""Own outputs until writeback/transfer and account shared storage once.

	Callers relinquish mutation rights when inserting a tensor. A lease must stay
	open through GPU or communication completion, including asynchronous reads.
	Eviction follows removal of the corresponding directory replica.
	"""

	def __init__(self, max_bytes: int) -> None:
		if max_bytes < 0:
			raise ValueError("Buffer budget must be nonnegative")
		self.max_bytes = max_bytes
		self._lock = RLock()
		self._buffers: Dict[int, _Buffer] = {}
		self._storage_users: Dict[Tuple[torch.device, int], int] = {}
		self._storage_bytes: Dict[Tuple[torch.device, int], int] = {}
		self._bytes = 0
		self._sequence = 0

	@property
	def bytes_used(self) -> int:
		with self._lock:
			return self._bytes

	def put(self, tensor: Tensor, *, readiness: Optional[Readiness] = None) -> int:
		if tensor.layout != torch.strided:
			raise ValueError("Embedding buffers must use strided tensor storage")
		storage = tensor.untyped_storage()
		identity = (tensor.device, storage.data_ptr())
		size = storage.nbytes()
		with self._lock:
			if identity in self._storage_bytes and self._storage_bytes[identity] != size:
				raise ValueError("Published tensor storage must not be resized")
			additional = 0 if identity in self._storage_users else size
			if self._bytes + additional > self.max_bytes:
				raise MemoryError("Resident result budget exhausted; write back or transfer owned results")
			if readiness is None and tensor.is_cuda:
				with torch.cuda.device(tensor.device):
					event = torch.cuda.Event()
					event.record(torch.cuda.current_stream(tensor.device))
					readiness = event
			self._sequence += 1
			handle = self._sequence
			self._buffers[handle] = _Buffer(tensor.detach(), identity, readiness)
			self._storage_users[identity] = self._storage_users.get(identity, 0) + 1
			self._storage_bytes[identity] = size
			self._bytes += additional
			return handle

	def is_ready(self, handle: int) -> bool:
		with self._lock:
			readiness = self._buffers[handle].readiness
			return readiness is None or readiness.query()

	def acquire(self, handle: int, *, wait: bool = False) -> TensorLease:
		with self._lock:
			buffer = self._buffers[handle]
			if buffer.retired:
				raise BufferInUse("Tensor state is retired")
			buffer.readers += 1
		try:
			if buffer.readiness is not None:
				if wait:
					buffer.readiness.synchronize()
				elif not buffer.readiness.query():
					raise BufferNotReady(handle)
			return TensorLease(self, handle, buffer.tensor)
		except BaseException:
			self._release(handle)
			raise

	def _release(self, handle: int) -> None:
		with self._lock:
			buffer = self._buffers[handle]
			if buffer.readers <= 0:
				raise RuntimeError("Tensor has no active reader to release")
			buffer.readers -= 1
			if not buffer.readers and buffer.retired:
				self._remove(handle)

	def retire(self, handle: int) -> None:
		"""Retire an unreachable state after directory removal; drain old readers."""
		with self._lock:
			buffer = self._buffers[handle]
			buffer.retired = True
			if not buffer.readers:
				self._remove(handle)

	def retire_all(self) -> None:
		"""Drop a failed endpoint's ownership while existing leases drain."""
		with self._lock:
			for handle in list(self._buffers):
				self.retire(handle)

	def mark_backed_up(self, handle: int) -> None:
		"""Called after control-plane acknowledgment of a live replacement copy."""
		with self._lock:
			self._buffers[handle].backed_up = True

	def mark_published(self, handle: int) -> None:
		self.mark_published_many((handle,))

	def mark_published_many(self, handles: Sequence[int]) -> None:
		"""Publish related outputs only after every buffer passes its readiness fence."""
		with self._lock:
			buffers = [(handle, self._buffers[handle]) for handle in handles]
			for handle, buffer in buffers:
				if buffer.retired:
					raise BufferInUse("Cannot publish retired tensor state")
				if not self.is_ready(handle):
					raise BufferNotReady(handle)
			for _, buffer in buffers:
				buffer.published = True

	def abandon(self, handle: int) -> None:
		"""Release a cancelled output that was never published to consumers."""
		with self._lock:
			buffer = self._buffers[handle]
			if buffer.published:
				raise UnflushedBuffer("Published state requires acknowledged ownership transfer")
			if buffer.readers:
				raise BufferInUse("Cannot abandon a tensor with active consumers")
			self._remove(handle)

	def evict(self, handle: int, *, authorize: Optional[Callable[[], object]] = None) -> None:
		with self._lock:
			buffer = self._buffers[handle]
			if not buffer.backed_up:
				raise UnflushedBuffer("Cannot evict the only owned copy")
			if buffer.readers:
				raise BufferInUse("Cannot evict a tensor with active consumers")
			if authorize is not None:
				authorize()
			self._remove(handle)

	def _remove(self, handle: int) -> None:
		buffer = self._buffers.pop(handle)
		identity = buffer.storage
		self._storage_users[identity] -= 1
		if not self._storage_users[identity]:
			self._bytes -= self._storage_bytes.pop(identity)
			del self._storage_users[identity]
