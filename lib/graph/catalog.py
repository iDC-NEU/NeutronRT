"""Compact vertex/layer indirection into versioned embedding chunks."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Callable, ContextManager, Dict, Iterator, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor

from lib.graph.ownership import EmbeddingDirectory, MissingState, StateKey


def _group_positions(identifiers: Tensor, positions: Optional[Tensor] = None) -> Iterator[Tuple[int, Tensor]]:
	"""Group row positions once, preserving query order within every chunk."""
	if not identifiers.numel():
		return
	order = identifiers.argsort(stable=True)
	keys, counts = identifiers[order].unique_consecutive(return_counts=True)
	for key, group in zip(keys.tolist(), order.split(counts.tolist()), strict=True):
		yield key, positions[group] if positions is not None else group


@dataclass(frozen=True)
class RowSelection:
	key: StateKey
	positions: Tensor
	rows: Tensor


@dataclass(frozen=True)
class PriorRows:
	version: int
	layer: int
	nodes: Tensor
	available: Tensor
	selections: Tuple[RowSelection, ...]


@dataclass
class _Chunk:
	key: StateKey
	nodes: Tensor
	users: int = 0
	published: bool = False


class EmbeddingCatalog:
	"""Dense integer metadata; embedding values remain with data-plane owners.

	Reserved node tensors are immutable task metadata and are retained by reference.
	The coordinator fences graph mutation while work at a version is executing.
	"""

	def __init__(self, nodes: int, layers: int, directory: EmbeddingDirectory, *, kind: str = "h") -> None:
		if nodes < 0 or layers < 1:
			raise ValueError("Invalid catalog dimensions")
		if kind not in ("h", "a"):
			raise ValueError("Catalog kind must identify embeddings or aggregates")
		self.nodes = nodes
		self.layers = layers
		self.directory = directory
		self.kind = kind
		self._lock = RLock()
		self._chunks: Dict[int, _Chunk] = {}
		self._row_chunk = torch.full((layers, nodes), -1, dtype=torch.int64)
		self._row_offset = torch.zeros((layers, nodes), dtype=torch.int64)
		self._valid = torch.zeros((layers, nodes), dtype=torch.bool)
		self._version = 0
		self._previous_version: Optional[int] = None
		self._sequence = 0
		self._pending_layers: Set[int] = set()
		self._invalid_since: Optional[Tensor] = None

	def enable_staleness(self) -> None:
		"""Allocate only version metadata; never retain historical embedding matrices."""
		with self._lock:
			if self._invalid_since is not None:
				return
			if self._version or self._pending_layers:
				raise RuntimeError("Enable bounded staleness before graph updates begin")
			self._invalid_since = torch.full_like(self._row_chunk, -1)

	def _mark_dirty(self, layer: int, nodes: Tensor, version: int) -> None:
		if self._invalid_since is not None:
			first = nodes[self._invalid_since[layer - 1, nodes] < 0]
			self._invalid_since[layer - 1, first] = version
		self._valid[layer - 1, nodes] = False

	@property
	def version(self) -> int:
		with self._lock:
			return self._version

	@property
	def pending_layers(self) -> Tuple[int, ...]:
		with self._lock:
			return tuple(sorted(self._pending_layers))

	def _check(self, layer: int, nodes: Tensor, version: int) -> None:
		if version != self._version:
			raise ValueError("Graph version is no longer current")
		if not 0 <= layer <= self.layers:
			raise ValueError("Layer out of range")
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Node IDs must be one-dimensional CPU int64 metadata")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.nodes):
			raise ValueError("Node ID out of range")

	def begin_version(self, version: int, dirty: Mapping[int, Tensor], *, pending_layers: Sequence[int] = ()) -> None:
		with self._lock:
			if version <= self._version:
				raise ValueError("Graph versions must increase")
			if self._pending_layers:
				raise RuntimeError("Previous graph version analysis is incomplete")
			if any(not 1 <= layer <= self.layers for layer in pending_layers):
				raise ValueError("Pending analysis layer out of range")
			for layer, nodes in dirty.items():
				self._check(layer, nodes, self._version)
				if layer == 0:
					raise ValueError("Authoritative feature updates belong to storage")
			for layer, nodes in dirty.items():
				self._mark_dirty(layer, nodes, version)
			self._previous_version = self._version
			self._version = version
			self._pending_layers = set(pending_layers)

	def finish_analysis(self, layer: int, dirty: Tensor, version: int) -> None:
		"""Resolve one layer's validity while GPU work at earlier layers proceeds."""
		with self._lock:
			self._check(layer, dirty, version)
			if layer not in self._pending_layers:
				raise ValueError("Layer analysis is not pending")
			identifiers = self._row_chunk[layer - 1, dirty]
			for identifier in identifiers.unique().tolist():
				if identifier < 0 or self._chunks[identifier].key.version < version:
					self._mark_dirty(layer, dirty[identifiers == identifier], version)
			self._pending_layers.remove(layer)

	def reserve(self, layer: int, nodes: Tensor, version: int) -> StateKey:
		with self._lock:
			self._check(layer, nodes, version)
			if layer == 0 or nodes.unique().numel() != nodes.numel():
				raise ValueError("Output nodes must be unique and belong to a model layer")
			self._sequence += 1
			key = StateKey(version, layer, self._sequence, self.kind)
			self._chunks[key.chunk] = _Chunk(key, nodes.detach())
			return key

	def publish(self, key: StateKey) -> Tuple[StateKey, ...]:
		"""Install ready output rows; return superseded chunks for retirement."""
		with self._lock:
			if key.version != self._version:
				raise ValueError("Publication graph version is no longer current")
			chunk = self._chunks[key.chunk]
			if chunk.key != key:
				raise ValueError("Publication state key does not match its reservation")
			self.directory.locate(key)
			if chunk.published:
				return ()
			layer, nodes = key.layer - 1, chunk.nodes
			previous, counts = self._row_chunk[layer, nodes].unique(return_counts=True)
			retired = []
			for identifier, count in zip(previous.tolist(), counts.tolist(), strict=True):
				if identifier < 0:
					continue
				old = self._chunks[identifier]
				old.users -= count
				if not old.users:
					retired.append(old.key)
					del self._chunks[identifier]
			self._row_chunk[layer].index_fill_(0, nodes, key.chunk)
			self._row_offset[layer].index_copy_(0, nodes, torch.arange(nodes.numel()))
			self._valid[layer, nodes] = True
			if self._invalid_since is not None:
				self._invalid_since[layer, nodes] = -1
			chunk.users = nodes.numel()
			chunk.published = True
			if not chunk.users:
				retired.append(key)
				del self._chunks[key.chunk]
			return tuple(retired)

	def available(self, layer: int, nodes: Tensor, version: int) -> Tensor:
		with self._lock:
			self._check(layer, nodes, version)
			if layer == 0:
				return torch.ones_like(nodes, dtype=torch.bool)
			valid = self._valid[layer - 1, nodes]
			identifiers = self._row_chunk[layer - 1, nodes]
			for identifier in identifiers[valid].unique().tolist():
				if layer in self._pending_layers and self._chunks[identifier].key.version != version:
					valid[identifiers == identifier] = False
					continue
				try:
					self.directory.locate(self._chunks[identifier].key)
				except MissingState:
					valid[identifiers == identifier] = False
			return valid

	def selections(self, layer: int, nodes: Tensor, version: int) -> Tuple[RowSelection, ...]:
		return self.relaxed_selections(layer, nodes, version, max_staleness=0)

	@contextmanager
	def previous_rows(
		self,
		layer: int,
		nodes: Tensor,
		version: int,
		*,
		retain: Callable[[Sequence[StateKey]], ContextManager[None]],
	) -> Iterator[PriorRows]:
		"""Retain certified old rows before pending analysis invalidates this layer.

		Missing rows and already-replaced current-version rows are unavailable; the
		caller must rebase those dependencies. Node IDs are borrowed immutable
		metadata. Routing offsets and availability survive later row publication.
		"""
		with ExitStack() as lifetime:
			with self._lock:
				self._check(layer, nodes, version)
				if layer not in self._pending_layers or self._previous_version is None:
					raise ValueError("Old rows must be captured before pending layer analysis finishes")
				valid = self._valid[layer - 1, nodes]
				identifiers = self._row_chunk[layer - 1, nodes]
				groups = []
				for identifier, positions in _group_positions(identifiers[valid], valid.nonzero().flatten()):
					key = self._chunks[identifier].key
					if key.version >= version:
						valid[positions] = False
						continue
					try:
						self.directory.locate(key)
					except MissingState:
						valid[positions] = False
						continue
					groups.append(RowSelection(key, positions, self._row_offset[layer - 1, nodes[positions]]))
				# Acquire lifetimes while holding the catalog lock: a concurrent
				# publication cannot retire a key between snapshot and retention.
				lifetime.enter_context(retain(tuple(group.key for group in groups)))
				result = PriorRows(self._previous_version, layer, nodes.detach(), valid, tuple(groups))
			yield result

	def ages(self, layer: int, nodes: Tensor, version: int) -> Tensor:
		"""Age since first invalidation, rather than physical tensor creation."""
		with self._lock:
			self._check(layer, nodes, version)
			if layer == 0 or self._invalid_since is None:
				return torch.zeros_like(nodes)
			first = self._invalid_since[layer - 1, nodes]
			return torch.where(first >= 0, version - first + 1, 0)

	def servable(self, layer: int, nodes: Tensor, version: int, *, max_staleness: int) -> Tensor:
		with self._lock:
			if type(max_staleness) is not int or max_staleness < 0:
				raise ValueError("Staleness budget must be a nonnegative integer")
			valid = self.available(layer, nodes, version)
			if not max_staleness or layer == 0:
				return valid
			if self._invalid_since is None:
				raise ValueError("Bounded staleness is disabled")
			if layer in self._pending_layers:
				return valid
			identifiers = self._row_chunk[layer - 1, nodes]
			candidates = (~valid) & (identifiers >= 0) & (self.ages(layer, nodes, version) <= max_staleness)
			for identifier in identifiers[candidates].unique().tolist():
				try:
					self.directory.locate(self._chunks[identifier].key)
				except MissingState:
					candidates[identifiers == identifier] = False
			return valid | candidates

	def relaxed_selections(
		self, layer: int, nodes: Tensor, version: int, *, max_staleness: int
	) -> Tuple[RowSelection, ...]:
		with self._lock:
			if layer == 0:
				raise ValueError("Input feature rows are loaded directly from storage")
			if not self.servable(layer, nodes, version, max_staleness=max_staleness).all():
				raise MissingState((layer, version))
			identifiers = self._row_chunk[layer - 1, nodes]
			groups = []
			for identifier, positions in _group_positions(identifiers):
				rows = self._row_offset[layer - 1, nodes[positions]]
				groups.append(RowSelection(self._chunks[identifier].key, positions, rows))
			return tuple(groups)

	def current_row_count(self, key: StateKey) -> int:
		"""Count current references without scanning or materializing their node IDs."""
		with self._lock:
			chunk = self._chunks.get(key.chunk)
			return chunk.users if chunk is not None and chunk.key == key else 0

	def discard_layer(self, layer: int, version: int) -> Tuple[StateKey, ...]:
		"""Unpublish derived rows after their execution-plan consumers finish."""
		with self._lock:
			self._check(layer, torch.empty(0, dtype=torch.int64), version)
			if layer == 0:
				raise ValueError("Authoritative input features cannot be discarded")
			chunks = [chunk for chunk in self._chunks.values() if chunk.key.layer == layer and chunk.published]
			for chunk in chunks:
				current = chunk.nodes[self._row_chunk[layer - 1, chunk.nodes] == chunk.key.chunk]
				self._row_chunk[layer - 1, current] = -1
				self._valid[layer - 1, current] = False
				if self._invalid_since is not None:
					self._invalid_since[layer - 1, current] = -1
				del self._chunks[chunk.key.chunk]
			return tuple(chunk.key for chunk in chunks)

	def current_nodes(self, key: StateKey) -> Tensor:
		with self._lock:
			chunk = self._chunks.get(key.chunk)
			if chunk is None or chunk.key != key:
				return torch.empty(0, dtype=torch.int64)
			mask = self._row_chunk[key.layer - 1, chunk.nodes] == key.chunk
			return chunk.nodes[mask]

	def cancel(self, key: StateKey) -> None:
		with self._lock:
			chunk = self._chunks[key.chunk]
			if chunk.key != key or chunk.published:
				raise ValueError("Only an unpublished reservation can be cancelled")
			del self._chunks[key.chunk]
