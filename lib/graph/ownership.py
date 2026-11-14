"""Storage-side publication of immutable embedding locations.

Entries describe task-sized tensors, rather than allocating Python objects for
every vertex. Tensor readiness and lifetime are the producing node's contract.
"""

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class StateKey:
	version: int
	layer: int
	chunk: int
	kind: str = "h"

	def __post_init__(self) -> None:
		if min(self.version, self.layer, self.chunk) < 0 or self.kind not in ("h", "a"):
			raise ValueError("Invalid embedding state key")


@dataclass(frozen=True)
class Owner:
	rank: int
	generation: int


@dataclass(frozen=True)
class ComputeAttempt:
	key: StateKey
	owner: Owner
	number: int


@dataclass(frozen=True)
class Location:
	owner: Owner
	handle: int


@dataclass(frozen=True)
class CopyAttempt:
	compute: ComputeAttempt
	source: Location
	target: Owner
	number: int


class MissingState(KeyError):
	"""No live, published copy exists; the caller must wait or recompute."""


class LastReplica(RuntimeError):
	"""Normal eviction would discard the only published copy."""


@dataclass
class _Entry:
	attempt: ComputeAttempt
	locations: Dict[Owner, Location] = field(default_factory=dict)


class EmbeddingDirectory:
	"""Version/location metadata; publishing never transfers embedding bytes."""

	def __init__(self) -> None:
		self._lock = RLock()
		self._generations: Dict[int, int] = {}
		self._owners: Dict[int, Owner] = {}
		self._entries: Dict[StateKey, _Entry] = {}
		self._copies: Dict[int, CopyAttempt] = {}
		self._sequence = 0

	def register_owner(self, rank: int) -> Owner:
		with self._lock:
			if rank < 0 or rank in self._owners:
				raise ValueError("Owner rank must be nonnegative and not already live")
			generation = self._generations.get(rank, 0) + 1
			owner = Owner(rank, generation)
			self._generations[rank] = generation
			self._owners[rank] = owner
			return owner

	def begin(self, key: StateKey, owner: Owner) -> ComputeAttempt:
		with self._lock:
			if self._owners.get(owner.rank) != owner:
				raise ValueError("Compute owner is not live")
			entry = self._entries.get(key)
			if entry is not None and entry.locations:
				raise ValueError("State is already published")
			self._sequence += 1
			attempt = ComputeAttempt(key, owner, self._sequence)
			self._entries[key] = _Entry(attempt)
			return attempt

	def publish(self, attempt: ComputeAttempt, handle: int) -> bool:
		"""Accept a ready-buffer notification, including idempotent retries."""
		if handle < 0:
			raise ValueError("Buffer handle must be nonnegative")
		with self._lock:
			entry = self._entries.get(attempt.key)
			if entry is None or entry.attempt != attempt or self._owners.get(attempt.owner.rank) != attempt.owner:
				return False
			location = Location(attempt.owner, handle)
			previous = entry.locations.get(attempt.owner)
			if previous is not None and previous != location:
				raise ValueError("A published attempt cannot replace its buffer")
			entry.locations[attempt.owner] = location
			return True

	def locate(self, key: StateKey, *, preferred: Optional[Owner] = None) -> Location:
		with self._lock:
			entry = self._entries.get(key)
			if entry is None or not entry.locations:
				raise MissingState(key)
			if preferred is not None and preferred in entry.locations:
				return entry.locations[preferred]
			return min(entry.locations.values(), key=lambda location: location.owner.rank)

	def begin_copy(self, key: StateKey, target: Owner, *, source: Optional[Location] = None) -> CopyAttempt:
		with self._lock:
			if self._owners.get(target.rank) != target:
				raise ValueError("Copy target is not live")
			entry = self._entries.get(key)
			if entry is None or not entry.locations:
				raise MissingState(key)
			if target in entry.locations:
				raise ValueError("Copy target already owns this state")
			if source is None:
				source = self.locate(key)
			if entry.locations.get(source.owner) != source:
				raise MissingState(key)
			self._sequence += 1
			attempt = CopyAttempt(entry.attempt, source, target, self._sequence)
			self._copies[attempt.number] = attempt
			return attempt

	def complete_copy(self, attempt: CopyAttempt, handle: int) -> bool:
		if handle < 0:
			raise ValueError("Buffer handle must be nonnegative")
		with self._lock:
			entry = self._entries.get(attempt.compute.key)
			issued = self._copies.get(attempt.number)
			if issued is not None and issued != attempt:
				return False
			if issued == attempt:
				del self._copies[attempt.number]
			if entry is None or entry.attempt != attempt.compute:
				return False
			location = Location(attempt.target, handle)
			if entry.locations.get(attempt.target) == location:
				return True
			if (
				issued != attempt
				or self._owners.get(attempt.target.rank) != attempt.target
				or entry.locations.get(attempt.source.owner) != attempt.source
				or attempt.target in entry.locations
			):
				return False
			entry.locations[attempt.target] = location
			return True

	def abort_copy(self, attempt: CopyAttempt) -> bool:
		with self._lock:
			if self._copies.get(attempt.number) != attempt:
				return False
			del self._copies[attempt.number]
			return True

	def lose_owner(self, owner: Owner) -> Tuple[StateKey, ...]:
		"""Invalidate one worker instance and return states needing recomputation."""
		with self._lock:
			if self._owners.get(owner.rank) != owner:
				return ()
			del self._owners[owner.rank]
			missing = []
			for key, entry in self._entries.items():
				removed = entry.locations.pop(owner, None)
				if not entry.locations and (removed is not None or entry.attempt.owner == owner):
					missing.append(key)
			self._copies = {
				number: attempt
				for number, attempt in self._copies.items()
				if owner not in (attempt.source.owner, attempt.target)
			}
			return tuple(missing)

	def remove_replica(self, key: StateKey, location: Location) -> bool:
		with self._lock:
			entry = self._entries[key]
			if entry.locations.get(location.owner) != location:
				return False
			if len(entry.locations) == 1:
				raise LastReplica(key)
			del entry.locations[location.owner]
			self._copies = {
				number: attempt
				for number, attempt in self._copies.items()
				if attempt.compute.key != key or attempt.source != location
			}
			return True

	def forget(self, key: StateKey) -> Tuple[Location, ...]:
		"""Retire unreachable state after the scheduler releases its references."""
		with self._lock:
			entry = self._entries.pop(key, None)
			self._copies = {number: attempt for number, attempt in self._copies.items() if attempt.compute.key != key}
			return tuple(entry.locations.values()) if entry is not None else ()
