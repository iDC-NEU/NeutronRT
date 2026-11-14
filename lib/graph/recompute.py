"""Storage-side dependency planning shared by baseline and recovery execution.

The caller holds a graph read fence while planning. Load and publication callbacks
must serve the requested version or reject it; no graph/feature snapshot is copied.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, cast

import dgl
import torch
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from torch import Tensor

if TYPE_CHECKING:
	from lib.graph.neighborhood import NeighborhoodPolicy

Availability = Callable[[int, Tensor, int], Tensor]
Load = Callable[[int, Tensor, int], Tensor]
Publish = Callable[[int, Tensor, Tensor, int], None]


class PlanTooLarge(MemoryError):
	pass


def compile_block(graph: DGLGraph, targets: Tensor) -> DGLBlock:
	"""Let storage backends compile selected rows without centralizing topology."""
	compiler = getattr(graph, "compile_block", None)
	if compiler is not None:
		return cast(DGLBlock, compiler(targets))
	frontier = dgl.in_subgraph(graph, targets, relabel_nodes=False)
	block = dgl.to_block(frontier, dst_nodes=targets, include_dst_in_src=True)
	if "nrt" in frontier.edata:
		block.edata["nrt"] = frontier.edata["nrt"][block.edata[dgl.EID]]
	return cast(DGLBlock, block)


@dataclass(frozen=True)
class LayerPlan:
	layer: int
	block: DGLBlock
	sources: Tensor
	targets: Tensor
	source_out_degrees: Optional[Tensor] = None


@dataclass(frozen=True)
class RecomputePlan:
	version: int
	layer: int
	requested: Tensor
	targets: Tensor
	tasks: Tuple[LayerPlan, ...]


class RecomputePlanner:
	def __init__(
		self,
		graph: DGLGraph,
		num_layers: int,
		*,
		max_edges: Optional[int] = None,
		batch_size: int = 8192,
		source_degree_sensitive: bool = False,
		policy: Optional["NeighborhoodPolicy"] = None,
	) -> None:
		if graph.device.type != "cpu":
			raise ValueError("Graph analysis requires the storage node's CPU graph")
		if num_layers < 1 or batch_size < 1 or max_edges is not None and max_edges < 0:
			raise ValueError("Invalid recomputation layer count or edge budget")
		self.graph = graph
		self.num_layers = num_layers
		self.max_edges = max_edges
		self.batch_size = batch_size
		self.source_degree_sensitive = source_degree_sensitive
		from lib.graph.neighborhood import NeighborhoodPolicy

		self.policy = policy or NeighborhoodPolicy()

	def _blocks(self, layer: int, nodes: Tensor, version: int) -> Iterator[LayerPlan]:
		response_bytes = getattr(self.graph, "response_bytes", None)
		if response_bytes is not None or self.max_edges is not None:
			degrees = self.graph.in_degrees(nodes)
			if self.policy.approximate:
				degrees = degrees.clamp_max(self.policy.fanout)
			count = int(degrees.sum())
			if self.max_edges is not None and count > self.max_edges:
				raise PlanTooLarge("Recomputation dependencies exceed the plan edge budget")
			if response_bytes is not None and (nodes.numel() + count) * 8 > response_bytes:
				if nodes.numel() <= 1:
					raise PlanTooLarge("A single adjacency row exceeds the storage response budget")
				middle = nodes.numel() // 2
				yield from self._blocks(layer, nodes[:middle], version)
				yield from self._blocks(layer, nodes[middle:], version)
				return
		try:
			block = self.policy.block(self.graph, nodes, layer, version)
		except MemoryError:
			# Partition a response rejected by storage before allocation.
			if nodes.numel() <= 1:
				raise
			middle = nodes.numel() // 2
			yield from self._blocks(layer, nodes[:middle], version)
			yield from self._blocks(layer, nodes[middle:], version)
			return
		sources = cast(Tensor, block.srcdata[dgl.NID])
		degrees = None
		if self.source_degree_sensitive:
			degrees = block.out_degrees() if self.policy.approximate else self.graph.out_degrees(sources)
		yield LayerPlan(layer, block, sources, cast(Tensor, block.dstdata[dgl.NID]), degrees)

	@staticmethod
	def _available(callback: Availability, layer: int, nodes: Tensor, version: int) -> Tensor:
		mask = callback(layer, nodes, version)
		if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.shape != nodes.shape:
			raise ValueError("Availability must return one CPU boolean per requested node")
		return mask

	def plan(
		self, nodes: Tensor, layer: int, version: int, *, available: Optional[Availability] = None
	) -> RecomputePlan:
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Query node IDs must be a one-dimensional CPU int64 tensor")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.graph.num_nodes()):
			raise ValueError("Query node ID out of range")
		if not 1 <= layer <= self.num_layers or version < 0:
			raise ValueError("Invalid requested layer or graph version")
		targets = nodes.unique(sorted=True)
		required = targets
		tasks: List[LayerPlan] = []
		edges = 0
		for current in range(layer, 0, -1):
			if not required.numel():
				break
			missing = required
			if available is not None and self.policy.reuse_intermediates:
				missing = required[~self._available(available, current, required, version)]
			if not missing.numel():
				break
			degrees_in = self.graph.in_degrees(missing)
			edges += int((degrees_in.clamp_max(self.policy.fanout) if self.policy.approximate else degrees_in).sum())
			if self.max_edges is not None and edges > self.max_edges:
				raise PlanTooLarge("Recomputation dependencies exceed the plan edge budget")
			compiled = [
				task for chunk in missing.split(self.batch_size) for task in self._blocks(current, chunk, version)
			]
			tasks.extend(compiled)
			required = torch.cat([task.sources for task in compiled]).unique(sorted=True)
			if current == 1 and available is not None and not self._available(available, 0, required, version).all():
				raise KeyError("Authoritative input features are unavailable for the requested version")
		return RecomputePlan(version, layer, nodes, targets, tuple(reversed(tasks)))

	def plans(
		self, nodes: Tensor, layer: int, version: int, *, available: Optional[Availability] = None
	) -> Iterator[RecomputePlan]:
		"""Stream bounded request plans; consumers execute each before requesting more.

		Row counts alone cannot bound skewed adjacency or multilayer dependencies.
		Keep the edge limit per live plan and recheck availability after consumers
		publish earlier results. A single destination that cannot fit still fails.
		"""
		try:
			plan = self.plan(nodes, layer, version, available=available)
		except PlanTooLarge:
			targets = nodes.unique(sorted=True)
			if targets.numel() <= 1:
				raise
		else:
			yield plan
			return
		# Leave the exception scope before recursing so rejected blocks and their
		# traceback do not remain resident while smaller plans execute.
		middle = targets.numel() // 2
		yield from self.plans(targets[:middle], layer, version, available=available)
		yield from self.plans(targets[middle:], layer, version, available=available)


@torch.inference_mode()
def execute_recomputation(
	plan: RecomputePlan, *, load: Load, apply: Callable[[LayerPlan, Tensor], Tensor], publish: Publish
) -> None:
	"""Execute the dependency DAG; apply/publish may retain results on a GPU."""
	for task in plan.tasks:
		inputs = load(task.layer - 1, task.sources, plan.version)
		if inputs.ndim != 2 or inputs.shape[0] != task.sources.numel():
			raise ValueError("Input rows do not match the planned source nodes")
		output = apply(task, inputs)
		if output.ndim != 2 or output.shape[0] != task.targets.numel():
			raise ValueError("Output rows do not match the planned destination nodes")
		publish(task.layer, task.targets, output, plan.version)
