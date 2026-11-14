"""Aggregate-only storage preserves GraphSAGE's self and neighbor terms."""

import dgl
import pytest
import torch

from lib.graph.deltas import NeighborUpdate, RoutedRows, update_neighbor_sums
from lib.graph.recompute import LayerPlan, compile_block
from lib.models.aggregates import full_sage_state, reconstruct_sage
from lib.models.spec import ModelSpec


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("precision", [torch.float32, torch.float64])
@torch.inference_mode()
def test_selected_aggregate_only_rows_match_dgl_with_self_bias_and_parallel_edges(projected, precision):
	cuda_was_initialized = torch.cuda.is_initialized()
	model = ModelSpec("sage", 5, 7, 3, layers=3, seed=31).build()
	model.layers[1].activation = torch.tanh
	model.layers[1].norm = torch.nn.functional.normalize
	graph = dgl.graph(([0, 0, 1, 1, 2, 3], [1, 1, 2, 0, 2, 3]), num_nodes=6)
	features = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 10
	selected = torch.tensor([5, 2, 0, 1, 5])
	states = []
	current = features
	for index in range(3):
		u, v = graph.edges()
		values = model.layers[index].fc_neigh(current) if projected else current
		sums = torch.zeros(6, values.shape[1], dtype=precision)
		sums.index_add_(0, v, values[u].to(precision))
		states.append(NeighborUpdate(sums[selected], graph.in_degrees(selected)))
		current = model(index, graph, current)
		actual = reconstruct_sage(model, features[selected], states, projected=projected)
		torch.testing.assert_close(actual, current[selected], atol=2e-6, rtol=2e-6)
	assert torch.cuda.is_initialized() == cuda_was_initialized


@torch.inference_mode()
def test_equal_neighbor_sums_do_not_erase_different_self_inputs():
	model = ModelSpec("sage", 2, 2, 2, seed=7).build()
	with torch.no_grad():
		model.layers[0].fc_self.weight.copy_(torch.eye(2))
		model.layers[0].fc_self.bias.zero_()
	state = NeighborUpdate(torch.zeros(2, 2), torch.zeros(2, dtype=torch.int64))
	actual = reconstruct_sage(model, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [state])
	torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_reconstruction_rejects_incomplete_or_incompatible_state_before_execution():
	model = ModelSpec("sage", 4, 5, 3).build()
	features = torch.ones(2, 4)
	state = NeighborUpdate(torch.zeros(2, 4), torch.ones(2, dtype=torch.int64))
	with pytest.raises(ValueError, match="requested model layer"):
		reconstruct_sage(model, features, [])
	with pytest.raises(ValueError, match="layer width"):
		reconstruct_sage(model, features, [state], projected=True)
	with pytest.raises(ValueError, match="input width"):
		reconstruct_sage(model, features[:, :2], [state])
	with pytest.raises(ValueError, match="eval-mode"):
		reconstruct_sage(model.train(), features, [state])
	model.eval()
	model.layers[0].fc_neigh = torch.nn.Linear(4, 5, bias=True)
	projected = NeighborUpdate(torch.zeros(2, 5), state.degrees)
	with pytest.raises(ValueError, match="bias-free"):
		reconstruct_sage(model, features, [projected], projected=True)


@pytest.mark.parametrize("projected", [False, True])
@torch.inference_mode()
def test_aggregate_only_state_survives_repeated_feature_and_edge_updates(projected):
	generator = torch.Generator().manual_seed(57)
	model = ModelSpec("sage", 4, 5, 3, seed=93).build()
	adjacency = torch.randint(0, 3, (9, 9), generator=generator)
	features = torch.randn(9, 4, generator=generator)

	def graph_for(adjacency):
		target, source = adjacency.nonzero(as_tuple=True)
		count = adjacency[target, source]
		return dgl.graph((source.repeat_interleave(count), target.repeat_interleave(count)), num_nodes=9)

	def project(layer, values):
		return model.layers[layer].fc_neigh(values) if projected else values

	def message(adjacency, values):
		target, source = adjacency.nonzero(as_tuple=True)
		return RoutedRows(values, source, target, adjacency[target, source])

	initial_graph = graph_for(adjacency)
	initial_first = model(0, initial_graph, features)
	states = [
		NeighborUpdate(adjacency.float() @ project(index, values), adjacency.sum(1))
		for index, values in enumerate((features, initial_first))
	]
	del initial_first, initial_graph  # Persistent model state consists only of aggregates and features.
	for step in range(50):
		new_adjacency = adjacency.clone()
		changed = torch.randperm(9, generator=generator)[:3]
		new_adjacency[changed] = torch.randint(0, 3, (3, 9), generator=generator)
		if step % 5 == 0:
			new_adjacency[changed[0]] = 0
		new_features = features.clone()
		new_features[changed] = torch.randn(3, 4, generator=generator)
		old_first = reconstruct_sage(model, features, states[:1], projected=projected)
		first = update_neighbor_sums(
			states[0].sums,
			states[0].degrees,
			message(new_adjacency, project(0, new_features - features)),
			message(new_adjacency - adjacency, project(0, features)),
			chunk_edges=3,
		)
		new_first = reconstruct_sage(model, new_features, [first], projected=projected)
		second = update_neighbor_sums(
			states[1].sums,
			states[1].degrees,
			message(new_adjacency, project(1, new_first - old_first)),
			message(new_adjacency - adjacency, project(1, old_first)),
			chunk_edges=3,
		)
		states, features, adjacency = [first, second], new_features, new_adjacency
		actual = reconstruct_sage(model, features, states, projected=projected)
		graph = graph_for(adjacency)
		expected = model(1, graph, model(0, graph, features))
		torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("empty", [False, True])
@torch.inference_mode()
def test_full_neighbor_initialization_returns_projected_sums_and_exact_output(empty):
	model = ModelSpec("sage", 4, 5, 3).build()
	graph = dgl.graph(([], []) if empty else ([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	current = features
	for index in range(2):
		targets = torch.tensor([4, 1, 0, 3])
		block = compile_block(graph, targets)
		sources = block.srcdata[dgl.NID]
		task = LayerPlan(index + 1, block, sources, targets)
		actual, sums = full_sage_state(model, task, current[sources])
		projected = model.layers[index].fc_neigh(current)
		expected_sums = torch.zeros_like(projected)
		u, v = graph.edges()
		expected_sums.index_add_(0, v, projected[u])
		torch.testing.assert_close(sums, expected_sums[targets])
		current = model(index, graph, current)
		torch.testing.assert_close(actual, current[targets], atol=2e-6, rtol=2e-6)
		assert "_nrt_projected" not in block.srcdata and "_nrt_sum" not in block.dstdata


@pytest.mark.parametrize("projected", [False, True])
@torch.inference_mode()
def test_cached_lower_embedding_can_start_reconstruction_at_its_layer(projected):
	model = ModelSpec("sage", 4, 5, 3, layers=3).build()
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	nodes = torch.tensor([4, 0, 2, 0])
	cached = model(0, graph, features)
	current = cached
	states = []
	for index in (1, 2):
		values = model.layers[index].fc_neigh(current) if projected else current
		sums = torch.zeros_like(values)
		u, v = graph.edges()
		sums.index_add_(0, v, values[u])
		states.append(NeighborUpdate(sums[nodes], graph.in_degrees(nodes)))
		current = model(index, graph, current)
	actual = reconstruct_sage(model, cached[nodes], states, projected=projected, start_layer=1)
	torch.testing.assert_close(actual, current[nodes], atol=2e-6, rtol=2e-6)
	with pytest.raises(ValueError, match="start layer"):
		reconstruct_sage(model, features[nodes], states, start_layer=3)
