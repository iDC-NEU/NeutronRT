"""Projected sparse model updates agree with independent full DGL inference."""

import dgl
import pytest
import torch

from lib.graph.delta_capture import capture_edges
from lib.graph.delta_plan import compile_delta_routes
from lib.graph.deltas import NeighborUpdate
from lib.models.aggregates import delta_sage_state
from lib.models.spec import ModelSpec


def _sum(graph, model, layer, inputs, dtype):
	projected = model.layers[layer].fc_neigh(inputs)
	result = torch.zeros((graph.num_nodes(), projected.shape[1]), dtype=dtype)
	u, v = graph.edges()
	result.index_add_(0, v, projected[u].to(dtype))
	return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.inference_mode()
def test_projected_sparse_updates_follow_three_layers_through_feature_and_parallel_edge_changes(dtype):
	model = ModelSpec("sage", 4, 5, 3, layers=3).build()
	graph = dgl.graph(([0, 0, 1, 2, 3, 4], [1, 1, 2, 0, 3, 1]), num_nodes=7)
	generator = torch.Generator().manual_seed(67)
	features = torch.randn(7, 4, generator=generator)
	h = [features.clone()]
	sums = []
	for layer in range(3):
		sums.append(_sum(graph, model, layer, h[-1], dtype))
		h.append(model(layer, graph, h[-1]))
	for step in range(24):
		old_h = h
		old_degree = graph.in_degrees()
		changed = torch.tensor([step % 7, (step + 2) % 7]).unique()
		features[changed] += torch.randn(changed.numel(), 4, generator=generator) * 0.2
		u, v = torch.tensor([step % 7, step % 7]), torch.tensor([(step + 1) % 7] * 2)
		delete = step % 3 == 2
		edges = capture_edges(graph, u, v, delete=delete)
		if delete:
			present = graph.has_edges_between(u, v)
			if present.any():
				_, _, ids = graph.edge_ids(u[present], v[present], return_uv=True)
				graph.remove_edges(ids.unique())
		else:
			graph.add_edges(u, v)
		h = [features.clone()]
		reference = features
		for layer in range(3):
			plan = compile_delta_routes(graph, changed, edges.sources, edges.targets, edges.weights, max_edges=512)
			before = sums[layer].clone()
			output, state = delta_sage_state(
				model,
				layer + 1,
				h[-1][plan.targets],
				NeighborUpdate(sums[layer][plan.targets], old_degree[plan.targets]),
				plan.inputs.bind(h[-1][plan.inputs.value_nodes] - old_h[layer][plan.inputs.value_nodes]),
				plan.edges.bind(old_h[layer][plan.edges.value_nodes]),
				chunk_edges=3,
			)
			expected_sum = _sum(graph, model, layer, reference, dtype)
			reference = model(layer, graph, reference)
			current = old_h[layer + 1].clone()
			current[plan.targets] = output
			sums[layer][plan.targets] = state.sums
			unchanged = torch.ones(7, dtype=torch.bool)
			unchanged[plan.targets] = False
			torch.testing.assert_close(sums[layer][unchanged], before[unchanged], rtol=0, atol=0)
			torch.testing.assert_close(sums[layer], expected_sum, atol=2e-5, rtol=2e-5)
			torch.testing.assert_close(current, reference, atol=2e-5, rtol=2e-5)
			torch.testing.assert_close(state.degrees, graph.in_degrees(plan.targets))
			h.append(current)
			changed = plan.targets
