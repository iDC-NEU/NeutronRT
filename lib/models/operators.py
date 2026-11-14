"""Run native DGL operators with the full-neighbor context they require."""

from typing import Any, cast

import dgl.nn as dglnn
from dgl.heterograph import DGLBlock
from torch import Tensor, nn

from lib.graph.recompute import LayerPlan


class _SourceDegreeContext:
	"""Override degree reads without mutating a block or changing native kernels."""

	def __init__(self, block: DGLBlock, degrees: Tensor) -> None:
		self._block = block
		self._degrees = degrees

	def __getattr__(self, name: str) -> Any:
		return getattr(self._block, name)

	def out_degrees(self) -> Tensor:
		return self._degrees


def full_neighbor_forward(layer: nn.Module, task: LayerPlan, features: Tensor) -> Tensor:
	graph = task.block
	if isinstance(layer, dglnn.GraphConv) and layer._norm in ("left", "both"):
		degrees = task.source_out_degrees
		if degrees is None or degrees.ndim != 1 or degrees.numel() != task.sources.numel():
			raise ValueError("GraphConv requires full-graph source out-degrees for block execution")
		return cast(Tensor, layer(_SourceDegreeContext(graph, degrees), features))
	return cast(Tensor, layer(graph, features))
