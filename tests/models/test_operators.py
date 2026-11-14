from dataclasses import replace

import dgl
import dgl.nn as dglnn
import pytest
import torch

from lib.graph.recompute import RecomputePlanner
from lib.models.operators import full_neighbor_forward


@pytest.mark.parametrize("norm", ["left", "both", "right", "none"])
def test_graphconv_block_uses_full_graph_degrees_without_mutating_the_block(norm):
	graph = dgl.graph(([0, 0, 0, 1, 2], [0, 1, 2, 2, 1]), num_nodes=3)
	features = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
	layer = dglnn.GraphConv(4, 3, norm=norm, allow_zero_in_degree=True).eval()
	task = RecomputePlanner(graph, 1, source_degree_sensitive=True).plan(torch.tensor([1]), 1, 0).tasks[0]
	local = task.block.out_degrees().clone()
	with torch.inference_mode():
		actual = full_neighbor_forward(layer, task, features[task.sources])
		expected = layer(graph, features)[task.targets]
	torch.testing.assert_close(actual, expected)
	torch.testing.assert_close(task.block.out_degrees(), local)
	if norm in ("left", "both"):
		with pytest.raises(ValueError, match="out-degrees"):
			full_neighbor_forward(layer, replace(task, source_out_degrees=None), features[task.sources])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a visible GPU")
def test_gpu_graphconv_preserves_global_source_normalization():
	graph = dgl.graph(([0, 0, 0, 1, 2], [0, 1, 2, 2, 1]), num_nodes=3)
	features = torch.arange(12, dtype=torch.float32, device="cuda:0").reshape(3, 4) / 10
	layer = dglnn.GraphConv(4, 3, norm="both", allow_zero_in_degree=True).eval().cuda()
	task = RecomputePlanner(graph, 1, source_degree_sensitive=True).plan(torch.tensor([1]), 1, 0).tasks[0]
	task = replace(task, block=task.block.to("cuda:0"))
	with torch.inference_mode():
		actual = full_neighbor_forward(layer, task, features[task.sources.to("cuda:0")])
		expected = layer(graph.to("cuda:0"), features)[task.targets.to("cuda:0")]
	torch.testing.assert_close(actual, expected)
