"""Versioned locations with remote GPU primaries and deferred CPU replicas."""

from contextlib import ExitStack, contextmanager
from typing import Dict, Iterator, Sequence, Tuple

import torch
from torch import Tensor

from lib.graph.catalog import RowSelection
from lib.graph.data_plane import LocalEmbeddingPlane, RejectedPublication
from lib.graph.ownership import ComputeAttempt, CopyAttempt, Location, MissingState, Owner, StateKey
from lib.graph.serving import select_rows
from lib.runtime.protocol import InputChunk
from lib.runtime.transport import OwnedTransport, ReadyResult


class RemoteEmbeddingPlane(LocalEmbeddingPlane):
	"""Keep the existing directory/writeback contract without master GPU tensors.

	The coordinator owns only its bounded CPU storage endpoint. Remote workers own
	their resident buffers. Query reads gather selected rows; background writeback
	creates a full CPU replica only after its receive and generation checks finish.
	"""

	def __init__(self, transport: OwnedTransport, *, storage_bytes: int, max_pending: int = 4) -> None:
		super().__init__(max_pending=max_pending)
		self.transport = transport
		self.storage = self.add_node(0, "cpu", storage_bytes)
		self.remotes = dict(transport.endpoints)
		self.discard_embeddings = False
		self._retained_states: Dict[StateKey, int] = {}
		self._pending_retire: set[StateKey] = set()
		self._shapes: Dict[StateKey, Tuple[int, int]] = {}
		for owner in self.remotes:
			if self.directory.register_owner(owner.rank) != owner:
				raise ValueError("Remote worker generation differs from the storage directory")

	def publish_result(self, attempt: ComputeAttempt, result: ReadyResult) -> Location:
		if (result.version, result.layer, result.chunk) != (attempt.key.version, attempt.key.layer, attempt.key.chunk):
			raise ValueError("Ready result does not match its state key")
		location = Location(attempt.owner, result.handle)
		if not self.directory.publish(attempt, result.handle):
			self.transport.retire(location)
			raise RejectedPublication("Remote compute attempt is no longer current")
		with self._lock:
			self._shapes[attempt.key] = result.shape
		return location

	def _copy(self, attempt: CopyAttempt) -> Location:
		if attempt.target != self.storage.owner:
			raise ValueError("Deferred remote writeback currently targets the storage CPU")
		handle = None
		accepted = False
		try:
			with self.transport.locked(attempt.source.owner):
				if self.directory.locate(attempt.compute.key, preferred=attempt.source.owner) != attempt.source:
					raise RejectedPublication("Writeback source is no longer current")
				with self._lock:
					rows, width = self._shapes[attempt.compute.key]
				pending = self.transport.read(attempt.source, torch.arange(rows), width)
			values = pending.result()
			handle = self.storage.store.put(values)
			with self.storage.store.acquire(handle, wait=True):
				if not self.directory.complete_copy(attempt, handle):
					raise RejectedPublication("Remote writeback attempt is no longer current")
				self.storage.store.mark_published(handle)
				self.storage.store.mark_backed_up(handle)
				accepted = True
			return Location(self.storage.owner, handle)
		finally:
			self.directory.abort_copy(attempt)
			if handle is not None and not accepted:
				self.storage.store.abandon(handle)

	def read_rows(self, key: StateKey, rows: Tensor, stack: ExitStack) -> Tensor:
		location = self.directory.locate(key, preferred=self.storage.owner)
		if location.owner == self.storage.owner:
			values = stack.enter_context(self.storage.store.acquire(location.handle, wait=True))
			return select_rows(values, rows)
		with self.transport.locked(location.owner):
			if self.directory.locate(key, preferred=location.owner) != location:
				raise MissingState(key)
			with self._lock:
				_, width = self._shapes[key]
			pending = self.transport.read(location, rows, width)
		return pending.result()

	def supply_inputs(
		self, inputs: Sequence[InputChunk], target: Owner, width: int, stack: ExitStack, *, gather: bool = False
	) -> None:
		"""Keep current-owner routing and optional gathering below model scheduling."""
		if not gather:
			for item in inputs:
				if item.location.owner == target:
					continue
				if item.location.owner == self.storage.owner:
					values = stack.enter_context(self.storage.store.acquire(item.location.handle, wait=True))
					self.transport.send_cpu_input(target, select_rows(values, item.rows))
				elif item.sender == 0:
					self.transport.send_cpu_input(target, self.transport.read(item.location, item.rows, width).result())
				else:
					self.transport.fetch_peer(item.location, item.rows, width, target)
			return
		by_sender: Dict[int, list] = {}
		for item in inputs:
			sender = item.location.owner.rank if item.sender is None else item.sender
			by_sender.setdefault(sender, []).append(item)
		for sender, groups in by_sender.items():
			if sender == target.rank:
				continue
			if sender == 0:
				self.transport.send_cpu_input(target, self._gather_root_inputs(groups, width, stack))
			else:
				self.transport.fetch_peer_many([(item.location, item.rows) for item in groups], width, target)

	def _gather_root_inputs(self, groups: Sequence[InputChunk], width: int, stack: ExitStack) -> Tensor:
		parts: Dict[int, Tensor] = {}
		remote: Dict[Owner, list] = {}
		for index, item in enumerate(groups):
			if item.location.owner == self.storage.owner:
				values = stack.enter_context(self.storage.store.acquire(item.location.handle, wait=True))
				parts[index] = select_rows(values, item.rows)
			else:
				remote.setdefault(item.location.owner, []).append((index, item))
		for entries in remote.values():
			values = self.transport.read_many([(item.location, item.rows) for _, item in entries], width).result()
			offset = 0
			for index, item in entries:
				parts[index] = values.narrow(0, offset, item.rows.numel())
				offset += item.rows.numel()
		return parts[0] if len(groups) == 1 else torch.cat([parts[index] for index in range(len(groups))])

	def gather_rows(self, selections: Sequence[RowSelection], width: int, stack: ExitStack) -> Tensor:
		if not selections:
			return torch.empty((0, width), dtype=torch.float32)
		groups = [
			InputChunk(
				self.directory.locate(group.key, preferred=self.storage.owner), group.rows, group.positions, sender=0
			)
			for group in selections
		]
		values = self._gather_root_inputs(groups, width, stack)
		positions = torch.cat([group.positions for group in selections])
		if torch.equal(positions, torch.arange(positions.numel())):
			return values
		output = torch.empty_like(values)
		output.index_copy_(0, positions, values)
		return output

	def make_room(self, owner: Owner, required: int) -> None:
		"""Apply backpressure only when retaining another owned output needs space."""
		node = self.remotes[owner]
		if required > node.max_bytes:
			raise MemoryError("A task output exceeds the worker resident budget; reduce the compute batch size")
		while node.bytes_used + required > node.max_bytes:
			with self._lock:
				keys = tuple(self._shapes)
			if self.discard_embeddings:
				keys = tuple(key for key in keys if key.kind == "h") + tuple(key for key in keys if key.kind != "h")
			victim = None
			for key in keys:
				try:
					location = self.directory.locate(key, preferred=owner)
				except MissingState:
					continue
				if location.owner == owner:
					victim = key
					break
			if victim is None:
				raise MemoryError("Worker has no published chunk available for writeback")
			if self.discard_embeddings and victim.kind == "h":
				self.retire(victim)
				continue
			self._fetch(victim, self.storage.owner)
			self.evict(victim, owner)

	def evict(self, key: StateKey, owner: Owner) -> None:
		if owner == self.storage.owner:
			super().evict(key, owner)
			return
		with self.transport.locked(owner):
			location = self.directory.locate(key, preferred=owner)
			if location.owner != owner:
				raise MissingState(key)
			self.directory.remove_replica(key, location)
			self.transport.retire(location)
			with self._lock:
				shape = self._shapes[key]
			node = self.remotes[owner]
			node.bytes_used = max(0, node.bytes_used - shape[0] * shape[1] * 4)

	def retire(self, key: StateKey) -> None:
		with self._lock:
			if key in self._retained_states:
				self._pending_retire.add(key)
				return
			locations = self.directory.forget(key)
			shape = self._shapes.pop(key, None)
		for location in locations:
			if location.owner == self.storage.owner:
				self.storage.store.retire(location.handle)
			elif location.owner in self.remotes:
				self.transport.retire(location)
				if shape is not None:
					node = self.remotes[location.owner]
					node.bytes_used = max(0, node.bytes_used - shape[0] * shape[1] * 4)

	@contextmanager
	def retain_states(self, keys: Sequence[StateKey]) -> Iterator[None]:
		"""Keep immutable old keys readable until dependent deltas finish.

		This retains metadata/lifetime, without copying values or preventing safe
		GPU-to-storage eviction. Retirement waits for the last nested consumer.
		"""
		unique = tuple(dict.fromkeys(keys))
		if self.discard_embeddings and any(key.kind == "h" for key in unique):
			raise ValueError("Aggregate-only mode must reconstruct historical h from aggregate anchors")
		with self._lock:
			if self._closed:
				raise RuntimeError("Data plane is closed")
			# Validate the whole set before retaining anything, including duplicates.
			for key in unique:
				self.directory.locate(key)
			for key in unique:
				self._retained_states[key] = self._retained_states.get(key, 0) + 1
		try:
			yield
		finally:
			retired = []
			with self._lock:
				for key in unique:
					remaining = self._retained_states[key] - 1
					if remaining:
						self._retained_states[key] = remaining
					else:
						del self._retained_states[key]
						if key in self._pending_retire:
							self._pending_retire.remove(key)
							retired.append(key)
			for key in retired:
				self.retire(key)

	@contextmanager
	def retain_aggregates(self, keys: Sequence[StateKey]) -> Iterator[None]:
		if any(key.kind != "a" for key in keys):
			raise ValueError("Aggregate anchors must name projected sum state")
		with self.retain_states(keys):
			yield

	def lose_owner(self, owner: Owner) -> Tuple[StateKey, ...]:
		if owner == self.storage.owner:
			return super().lose_owner(owner)
		with self._lock:
			missing = self.directory.lose_owner(owner)
			self.remotes.pop(owner, None)
			return missing
