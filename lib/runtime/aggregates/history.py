"""Coherent old aggregate prefixes for reconstructing selected historical h."""

from typing import TYPE_CHECKING, Tuple, cast

import torch
from torch import Tensor

from lib.graph.catalog import RowSelection
from lib.graph.ownership import MissingState, Owner
from lib.runtime.aggregates.delta_inputs import DeltaInputs
from lib.runtime.aggregates.delta_state import DeltaContext
from lib.runtime.aggregates.tables import InputTable

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


class HistoricalAggregates:
	def __init__(self, service: "DistributedEmbeddingService") -> None:
		if service.aggregate_catalog is None:
			raise ValueError("Historical reconstruction requires retained aggregates")
		self.service = service
		self.inputs = DeltaInputs(service)

	def _check(self, context: DeltaContext, layer: int) -> None:
		if context.state.active is not context or not context.enabled:
			raise MissingState("Historical inputs require their live graph-update context")
		if not 1 <= layer < len(self.service.widths) or any(
			prefix not in context.layers for prefix in range(1, layer + 1)
		):
			raise MissingState("Historical aggregate prefix analysis is incomplete")

	def prepare(self, context: DeltaContext, nodes: Tensor, layer: int) -> None:
		"""Compute only missing CURRENT sums outside each complete changed frontier."""
		self._check(context, layer)
		for prefix in range(1, layer + 1):
			selected = context.layers[prefix].aggregates.lookup(nodes)
			if (selected.covered & ~selected.available).any():
				raise MissingState("A changed prefix row has no certified old aggregate")
			if (~selected.covered).any():
				self.service._ensure_rows(nodes[~selected.covered], prefix)

	def tables(self, context: DeltaContext, nodes: Tensor, layer: int, target: Owner) -> Tuple[InputTable, ...]:
		"""Resolve locations after output/workspace admission can evict replicas."""
		self._check(context, layer)
		catalog = self.service.aggregate_catalog
		assert catalog is not None
		tables = [self.inputs.features_old(nodes, context.mutation.features)]
		for prefix in range(1, layer + 1):
			selected = context.layers[prefix].aggregates.lookup(nodes)
			if (selected.covered & ~selected.available).any():
				raise MissingState("A changed prefix row has no certified old aggregate")
			groups = list(selected.selections)
			outside = (~selected.covered).nonzero().flatten()
			if outside.numel():
				current = catalog.selections(prefix, nodes[outside], self.service.catalog.version)
				groups.extend(RowSelection(group.key, outside[group.positions], group.rows) for group in current)
			tables.append(
				InputTable(nodes.numel(), self.service.widths[prefix], chunks=self.inputs._routes(groups, target))
			)
		return tuple(tables)

	def degrees(self, context: DeltaContext, nodes: Tensor, layer: int) -> Tensor:
		self._check(context, layer)
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Historical degrees require CPU int64 node IDs")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.service.graph.num_nodes()):
			raise ValueError("Historical degree node is outside the graph")
		degrees = cast(Tensor, self.service.graph.in_degrees(nodes)).clone()
		edges = context.mutation.edges
		if edges.targets.numel() and nodes.numel():
			targets, indices = edges.targets.unique(sorted=True, return_inverse=True)
			changes = torch.zeros_like(targets)
			changes.index_add_(0, indices, edges.weights)
			positions = torch.searchsorted(targets, nodes).clamp_max(targets.numel() - 1)
			changed = targets[positions] == nodes
			degrees[changed] -= changes[positions[changed]]
		if (degrees < 0).any():
			raise RuntimeError("Historical structural corrections imply a negative degree")
		return degrees
