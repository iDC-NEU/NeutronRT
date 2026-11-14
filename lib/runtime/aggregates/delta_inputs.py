"""Resolve compact old/new inputs without whole-graph value snapshots."""

from typing import TYPE_CHECKING, Sequence, Tuple

import torch
from torch import Tensor

from lib.graph.anchors import AnchoredRows
from lib.graph.catalog import RowSelection
from lib.graph.delta_capture import FeatureSnapshot
from lib.graph.ownership import MissingState, Owner
from lib.graph.serving import select_rows
from lib.runtime.aggregates.tables import InputTable
from lib.runtime.protocol import InputChunk

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


class DeltaInputs:
	def __init__(self, service: "DistributedEmbeddingService") -> None:
		self.service = service
		self.plane = service.remote_plane

	def _routes(self, groups: Sequence[RowSelection], target: Owner) -> Tuple[InputChunk, ...]:
		result = []
		for group in groups:
			location = self.plane.directory.locate(group.key, preferred=target)
			proxy = not self.service.peer_fetch and location.owner not in (target, self.plane.storage.owner)
			result.append(InputChunk(location, group.rows, group.positions, sender=0 if proxy else None))
		return tuple(result)

	def current(self, layer: int, nodes: Tensor, target: Owner) -> InputTable:
		if layer == 0:
			return InputTable(nodes.numel(), self.service.widths[0], values=select_rows(self.service.features, nodes))
		groups = self.service.catalog.selections(layer, nodes, self.service.catalog.version)
		return InputTable(nodes.numel(), self.service.widths[layer], chunks=self._routes(groups, target))

	def old_sum(self, anchors: AnchoredRows, nodes: Tensor, target: Owner) -> InputTable:
		if any(group.key.kind != "a" for group in anchors.snapshot.selections):
			raise ValueError("Old sums must route aggregate keys")
		selected = anchors.lookup(nodes)
		if not selected.available.all():
			raise MissingState("Some old aggregate rows are unavailable")
		return InputTable(
			nodes.numel(), self.service.widths[anchors.snapshot.layer], chunks=self._routes(selected.selections, target)
		)

	def prepare_old(self, anchors: AnchoredRows, nodes: Tensor) -> None:
		"""A row outside the complete changed frontier is unchanged at this layer."""
		if any(group.key.kind != "h" for group in anchors.snapshot.selections):
			raise ValueError("Old inputs must route embedding keys")
		selected = anchors.lookup(nodes)
		if (selected.covered & ~selected.available).any():
			raise MissingState("A changed input has no certified old embedding")
		if (~selected.covered).any():
			self.service._ensure_rows(nodes[~selected.covered], anchors.snapshot.layer)

	def old(self, anchors: AnchoredRows, nodes: Tensor, target: Owner) -> InputTable:
		if any(group.key.kind != "h" for group in anchors.snapshot.selections):
			raise ValueError("Old inputs must route embedding keys")
		selected = anchors.lookup(nodes)
		if (selected.covered & ~selected.available).any():
			raise MissingState("A changed input has no certified old embedding")
		groups = list(selected.selections)
		outside = (~selected.covered).nonzero().flatten()
		if outside.numel():
			current = self.service.catalog.selections(
				anchors.snapshot.layer, nodes[outside], self.service.catalog.version
			)
			groups.extend(RowSelection(group.key, outside[group.positions], group.rows) for group in current)
		return InputTable(
			nodes.numel(), self.service.widths[anchors.snapshot.layer], chunks=self._routes(groups, target)
		)

	def features_old(self, nodes: Tensor, snapshot: FeatureSnapshot | None) -> InputTable:
		features = self.service.features
		if snapshot is None or not snapshot.nodes.numel():
			return InputTable(nodes.numel(), features.shape[1], values=select_rows(features, nodes))
		indices = torch.searchsorted(snapshot.nodes, nodes).clamp_max(snapshot.nodes.numel() - 1)
		changed = snapshot.nodes[indices] == nodes
		if changed.all():
			values = select_rows(snapshot.values, indices)
		elif changed.any():
			# Only this selected table is copied. Never overwrite a borrowed storage
			# feature view, and never recover old values by subtracting a float delta.
			values = torch.index_select(features, 0, nodes)
			values[changed] = snapshot.values[indices[changed]]
		else:
			values = select_rows(features, nodes)
		return InputTable(nodes.numel(), features.shape[1], values=values)
