"""Independent dense and DGL oracles for the sparse aggregate primitive."""

import dgl
import pytest
import torch

from lib.graph.deltas import RoutedRows, update_neighbor_sums
from lib.models.spec import ModelSpec


def _routes(adjacency, values, targets):
	"""Adjacency is target by source; values retain one row per source."""
	selected = adjacency[targets]
	destination, source = selected.nonzero(as_tuple=True)
	return RoutedRows(values, source, destination, selected[destination, source])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk", [1, 3, 8192])
def test_simultaneous_structure_and_input_changes_match_dense_multigraph(dtype, chunk):
	generator = torch.Generator().manual_seed(31)
	old = torch.randint(0, 4, (11, 11), generator=generator)
	new = torch.randint(0, 4, (11, 11), generator=generator)
	new[3] = 0  # Delete a whole neighborhood, including parallel/self edges.
	before = torch.randn(11, 14, generator=generator, dtype=dtype)[:, ::2]
	after = before.clone()
	after[[0, 3, 10]] += torch.randn(3, 7, generator=generator, dtype=dtype)
	targets = torch.tensor([7, 3, 0, 10])
	sums = (old.to(dtype) @ before)[targets]
	degrees = old.sum(1)[targets]
	original_sums, original_degrees = sums.clone(), degrees.clone()
	actual = update_neighbor_sums(
		sums,
		degrees,
		_routes(new, after - before, targets),
		_routes(new - old, before, targets),
		chunk_edges=chunk,
	)
	expected_sums = (new.to(dtype) @ after)[targets]
	tolerance = 3e-5 if dtype == torch.float32 else 3e-13
	torch.testing.assert_close(actual.sums, expected_sums, atol=tolerance, rtol=tolerance)
	torch.testing.assert_close(actual.degrees, new.sum(1)[targets], atol=0, rtol=0)
	torch.testing.assert_close(
		actual.mean(dtype=dtype),
		expected_sums / new.sum(1)[targets, None].clamp_min(1),
		atol=tolerance,
		rtol=tolerance,
	)
	assert torch.equal(sums, original_sums) and torch.equal(degrees, original_degrees)
	assert actual.sums.data_ptr() != sums.data_ptr() and actual.degrees.data_ptr() != degrees.data_ptr()
	assert torch.count_nonzero(actual.sums[1]) == 0


def test_feature_change_on_a_newly_inserted_edge_uses_the_new_input_once():
	old, new = torch.zeros(2, 2, dtype=torch.int64), torch.tensor([[0, 0], [1, 0]])
	before, after = torch.tensor([[2.0], [0.0]]), torch.tensor([[7.0], [0.0]])
	targets = torch.tensor([1])
	actual = update_neighbor_sums(
		torch.zeros(1, 1),
		torch.zeros(1, dtype=torch.int64),
		_routes(new, after - before, targets),
		_routes(new - old, before, targets),
	)
	assert actual.sums.item() == 7 and actual.degrees.item() == 1


@pytest.mark.parametrize("coalesced", [False, True])
@torch.inference_mode()
def test_repeated_and_coalesced_routes_are_equivalent_and_retry_is_immutable(coalesced):
	values = torch.tensor([[2.0, 5.0], [7.0, 11.0]])
	if coalesced:
		source, destination, weight = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([3, 2])
	else:
		source, destination, weight = torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 0, 0, 1, 1]), None
	inputs = RoutedRows(values, source, destination, weight)
	empty = torch.empty(0, dtype=torch.int64)
	edges = RoutedRows(values, empty, empty, empty)
	sums = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
	degrees = torch.tensor([3, 2])
	first = update_neighbor_sums(sums, degrees, inputs, edges)
	first.sums.add_(100)  # An abandoned replacement must not poison the retry.
	retry = update_neighbor_sums(sums, degrees, inputs, edges)
	torch.testing.assert_close(retry.sums, torch.tensor([[7.0, 18.0], [18.0, 28.0]]))
	torch.testing.assert_close(sums, torch.tensor([[1.0, 3.0], [4.0, 6.0]]))


def test_invalid_structure_correction_cannot_change_published_rows():
	sums, degrees = torch.ones(2, 3), torch.tensor([1, 0])
	empty = torch.empty(0, dtype=torch.int64)
	inputs = RoutedRows(torch.empty(0, 3), empty, empty)
	edges = RoutedRows(torch.ones(1, 3), torch.tensor([0]), torch.tensor([1]), torch.tensor([-1]))
	with pytest.raises(ValueError, match="negative degree"):
		update_neighbor_sums(sums, degrees, inputs, edges)
	assert torch.equal(sums, torch.ones(2, 3)) and torch.equal(degrees, torch.tensor([1, 0]))
	with pytest.raises(ValueError, match="outside"):
		update_neighbor_sums(sums, degrees, RoutedRows(torch.ones(1, 3), torch.tensor([2]), torch.tensor([0])), edges)
	with pytest.raises(ValueError, match="scratch bound"):
		update_neighbor_sums(sums, degrees, inputs, edges, chunk_edges=True)


def test_empty_selection_and_empty_neighborhood_remain_valid():
	empty = torch.empty(0, dtype=torch.int64)
	message = RoutedRows(torch.empty(0, 4), empty, empty, empty)
	actual = update_neighbor_sums(torch.empty(0, 4), empty, message, message)
	assert actual.mean().shape == (0, 4) and actual.degrees.shape == (0,)
	actual = update_neighbor_sums(torch.full((1, 4), 1e-8), torch.tensor([0]), message, message)
	assert torch.count_nonzero(actual.mean()) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.inference_mode()
def test_multilayer_sparse_updates_match_full_sage_over_a_long_mutation_trace(dtype):
	generator = torch.Generator().manual_seed(92)
	model = ModelSpec("sage", 5, 7, 3, seed=18).build()
	count = 13
	adjacency = torch.randint(0, 3, (count, count), generator=generator)
	features = torch.randn(count, 5, generator=generator)
	targets = torch.arange(count)
	layers = [features]
	aggregates = []
	for layer in range(2):
		sums = adjacency.to(dtype) @ layers[-1].to(dtype)
		aggregates.append(sums)
		mean = (sums / adjacency.sum(1).clamp_min(1)[:, None]).float()
		layers.append(model.from_aggrs(layer, layers[-1], mean))
	for step in range(80):
		updated_adjacency = adjacency.clone()
		changed = torch.randperm(count, generator=generator)[:3]
		updated_adjacency[changed] = torch.randint(0, 3, (3, count), generator=generator)
		if step % 9 == 0:
			updated_adjacency[changed[0]] = 0
		updated_features = features.clone()
		updated_features[changed] = torch.randn(3, 5, generator=generator)
		new_layers = [updated_features]
		for layer in range(2):
			old, new = layers[layer], new_layers[layer]
			actual = update_neighbor_sums(
				aggregates[layer],
				adjacency.sum(1),
				_routes(updated_adjacency, new.to(dtype) - old.to(dtype), targets),
				_routes(updated_adjacency - adjacency, old.to(dtype), targets),
				chunk_edges=7,
			)
			aggregates[layer] = actual.sums
			new_layers.append(model.from_aggrs(layer, new, actual.mean()))
		# Independent DGL execution, including parallel edges, self edges and isolated vertices.
		destination, source = updated_adjacency.nonzero(as_tuple=True)
		multiplicity = updated_adjacency[destination, source]
		graph = dgl.graph(
			(source.repeat_interleave(multiplicity), destination.repeat_interleave(multiplicity)), num_nodes=count
		)
		expected = model(1, graph, model(0, graph, updated_features))
		torch.testing.assert_close(new_layers[-1], expected, atol=3e-5, rtol=3e-5)
		features, adjacency, layers = updated_features, updated_adjacency, new_layers
