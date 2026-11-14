"""Bounded old-state lifetimes for one strict graph-update transition."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Sequence, Union

import torch
from torch import Tensor

from lib.graph.anchors import AnchoredRows
from lib.graph.catalog import PriorRows, RowSelection
from lib.graph.delta_capture import EdgeCorrections, FeatureSnapshot, capture_edges, capture_features
from lib.graph.ownership import StateKey
from lib.graph.recompute import PlanTooLarge

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


@dataclass(frozen=True)
class PendingDelta:
	features: Optional[FeatureSnapshot]
	edges: EdgeCorrections
	snapshot_bytes: int = 0
	enabled: bool = True


@dataclass(frozen=True)
class LayerAnchors:
	embeddings: AnchoredRows
	aggregates: AnchoredRows


class DeltaState:
	def __init__(
		self, service: "DistributedEmbeddingService", *, max_steps: int, anchor_bytes: int, row_refresh: bool = False
	) -> None:
		if type(max_steps) is not int or max_steps < 1 or type(anchor_bytes) is not int or anchor_bytes < 0:
			raise ValueError("Delta refresh steps must be positive and anchor bytes nonnegative")
		self.service, self.max_steps, self.anchor_bytes = service, max_steps, anchor_bytes
		if type(row_refresh) is not bool:
			raise ValueError("Row refresh must be a boolean")
		self.row_refresh = row_refresh
		self.pending: Optional[PendingDelta] = None
		self.active: Optional[DeltaContext] = None
		self._lock = RLock()
		self._depths: Dict[StateKey, Union[int, Tensor]] = {}

	def prepare_features(self, nodes: Tensor) -> None:
		empty = torch.empty(0, dtype=torch.int64)
		try:
			snapshot = capture_features(self.service.features, nodes, max_bytes=self.anchor_bytes)
		except PlanTooLarge:
			self.pending = PendingDelta(None, EdgeCorrections(empty, empty, empty), enabled=False)
			return
		self.pending = PendingDelta(
			snapshot,
			EdgeCorrections(empty, empty, empty),
			snapshot.nodes.numel() * (8 + self.service.features.shape[1] * self.service.features.element_size()),
		)

	def prepare_edges(self, sources: Tensor, targets: Tensor, *, delete: bool) -> None:
		self.pending = PendingDelta(None, capture_edges(self.service.graph, sources, targets, delete=delete))

	@contextmanager
	def updating(self) -> Iterator[None]:
		pending, self.pending = self.pending, None
		if pending is None:
			yield
			return
		if self.active is not None:
			raise RuntimeError("A sparse update context is already active")
		context = DeltaContext(self, pending)
		self.active = context
		try:
			yield
		finally:
			self.active = None
			try:
				context.close()
			finally:
				self.prune()

	def depth(self, keys: Sequence[StateKey]) -> int:
		with self._lock:
			depths = []
			for key in keys:
				value = self._depths.get(key, 0)
				depths.append((int(value.max()) if value.numel() else 0) if isinstance(value, Tensor) else value)
			return max(depths, default=0)

	def row_depths(self, groups: Sequence[RowSelection], rows: int) -> Tensor:
		"""Route integer metadata through the same old-row offsets as the sums."""
		output = torch.zeros(rows, dtype=torch.int64)
		with self._lock:
			for group in groups:
				values = self._depths.get(group.key, 0)
				output[group.positions] = values[group.rows] if isinstance(values, Tensor) else values
		return output

	@property
	def metadata_bytes(self) -> int:
		"""Live integer tensor storage only; excludes Python/allocator overhead."""
		with self._lock:
			return sum(value.untyped_storage().nbytes() for value in self._depths.values() if isinstance(value, Tensor))

	def record(self, key: StateKey, depth: Union[int, Tensor], obsolete: Sequence[StateKey] = ()) -> None:
		if isinstance(depth, Tensor):
			shape = self.service.remote_plane._shapes.get(key)
			if (
				not self.row_refresh
				or shape is None
				or key.kind != "a"
				or depth.device.type != "cpu"
				or depth.dtype != torch.int64
				or depth.shape != (shape[0],)
				or depth.numel()
				and (depth.min() < 0 or depth.max() > self.max_steps)
			):
				raise ValueError("Row depths must match a live aggregate chunk and its refresh limit")
			depth = depth.detach().clone()
		elif type(depth) is not int or not 0 <= depth <= self.max_steps or key.kind != "a":
			raise ValueError("Aggregate correction depth is outside its refresh limit")
		with self._lock:
			self._depths[key] = depth
			if self.active is None:
				for old in obsolete:
					self._depths.pop(old, None)

	def prune(self) -> None:
		catalog = self.service.aggregate_catalog
		assert catalog is not None
		with self._lock:
			self._depths = {key: depth for key, depth in self._depths.items() if catalog.current_row_count(key)}


class DeltaContext:
	def __init__(self, state: DeltaState, mutation: PendingDelta) -> None:
		self.state, self.mutation = state, mutation
		self.service = state.service
		self.plane = self.service.remote_plane
		self.version = self.service.catalog.version + 1
		self.owners = frozenset(self.plane.remotes)
		self.layers: Dict[int, LayerAnchors] = {}
		self._stack = ExitStack()
		self._lock = RLock()
		self._keys: set[StateKey] = set()
		self.anchor_bytes = mutation.snapshot_bytes
		self.disabled = not mutation.enabled

	@property
	def enabled(self) -> bool:
		with self._lock, self.plane._lock:
			if self.owners != frozenset(self.plane.remotes):
				self.disabled = True
			return not self.disabled and self.service.catalog.version == self.version

	def capture(self, layer: int, nodes: Tensor, version: int) -> None:
		with self._lock:
			if not self.enabled:
				return
			if version != self.version or layer in self.layers:
				raise ValueError("Sparse layer anchors must be captured once at their active update version")
			catalog = self.service.aggregate_catalog
			assert catalog is not None
			with ExitStack() as lifetime:
				if self.service.aggregate_only:
					# The complete changed frontier is still needed for route planning.
					# Its old h is reconstructed from sums, never retained or backed up.
					h = PriorRows(version - 1, layer, nodes, torch.zeros_like(nodes, dtype=torch.bool), ())
				else:
					h = lifetime.enter_context(
						self.service.catalog.previous_rows(layer, nodes, version, retain=self.plane.retain_states)
					)
				a = lifetime.enter_context(
					catalog.previous_rows(layer, nodes, version, retain=self.plane.retain_aggregates)
				)
				if h.version != version - 1 or a.version != version - 1:
					raise ValueError("Sparse aggregate anchors must certify the immediately preceding graph version")
				keys = {group.key for snapshot in (h, a) for group in snapshot.selections}
				with self.plane._lock:
					if any(key not in self.plane._shapes for key in keys):
						self.disabled = True
						return
					cost = sum(self.plane._shapes[key][0] * self.plane._shapes[key][1] * 4 for key in keys - self._keys)
				if self.anchor_bytes + cost > self.state.anchor_bytes:
					# Keep earlier contexts alive for any already-running layer task;
					# release only this unconsumed acquisition and use full recomputation.
					self.disabled = True
					return
				self.layers[layer] = LayerAnchors(AnchoredRows(h), AnchoredRows(a))
				self._keys.update(keys)
				self.anchor_bytes += cost
				self._stack.enter_context(lifetime.pop_all())

	def close(self) -> None:
		self._stack.close()
