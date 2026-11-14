"""Lazy, storage-side affected-node analysis and full-neighbor task construction."""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, cast

import dgl
import torch
from dgl import DGLGraph
from torch import Tensor

from lib.graph.recompute import LayerPlan, PlanTooLarge, compile_block


@dataclass(frozen=True)
class AffectedLayer:
	layer: int
	targets: Tensor


class AffectedPlanner:
	def __init__(
		self,
		graph: DGLGraph,
		layers: int,
		*,
		source_degree_sensitive: bool = False,
		expander: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
	) -> None:
		if graph.device.type != "cpu" or layers < 1:
			raise ValueError("Affected analysis requires a CPU storage graph and positive layer count")
		self.graph = graph
		self.layers = layers
		self.source_degree_sensitive = source_degree_sensitive
		self.expander = expander

	def _nodes(self, nodes: Optional[Tensor]) -> Tensor:
		if nodes is None:
			return torch.empty(0, dtype=torch.int64)
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Affected node IDs must be one-dimensional CPU int64 metadata")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.graph.num_nodes()):
			raise ValueError("Affected node ID out of range")
		return nodes

	def analyze(
		self,
		*,
		changed_features: Optional[Tensor] = None,
		edge_targets: Optional[Tensor] = None,
		degree_sources: Optional[Tensor] = None,
	) -> Iterator[AffectedLayer]:
		"""Advance one layer only when requested, so next-layer analysis can overlap GPU work.

		Edge targets participate at every layer because the propagation graph changes
		at every layer. Models using source-degree normalization also supply changed
		edge sources. Keeping the current frontier covers self/residual dependencies.
		"""
		frontier = torch.cat([self._nodes(changed_features), self._nodes(degree_sources)]).unique(sorted=True)
		edge_nodes = self._nodes(edge_targets)
		for layer in range(1, self.layers + 1):
			if self.expander is not None:
				frontier = self.expander(frontier, edge_nodes)
			else:
				neighbors = self.graph.out_edges(frontier)[1] if frontier.numel() else frontier
				frontier = torch.cat([frontier, neighbors, edge_nodes]).unique(sorted=True)
			yield AffectedLayer(layer, frontier)

	def block(self, layer: int, targets: Tensor, *, max_edges: Optional[int] = None) -> LayerPlan:
		targets = self._nodes(targets)
		if not 1 <= layer <= self.layers:
			raise ValueError("Invalid affected layer")
		if targets.unique().numel() != targets.numel():
			raise ValueError("A task must have unique output nodes")
		if max_edges is not None:
			if max_edges < 0:
				raise ValueError("Edge budget must be nonnegative")
			if int(self.graph.in_degrees(targets).sum()) > max_edges:
				raise PlanTooLarge("Affected task exceeds its edge budget")
		block = compile_block(self.graph, targets)
		sources = cast(Tensor, block.srcdata[dgl.NID])
		degrees = self.graph.out_degrees(sources) if self.source_degree_sensitive else None
		return LayerPlan(layer, block, sources, cast(Tensor, block.dstdata[dgl.NID]), degrees)
