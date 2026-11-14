"""Capture old values and delete-all multiplicity, without whole-state snapshots."""

import dgl
import pytest
import torch

from lib.graph.delta_capture import capture_edges, capture_features
from lib.graph.delta_plan import compile_delta_routes
from lib.graph.deltas import update_neighbor_sums
from lib.graph.recompute import PlanTooLarge


@pytest.mark.parametrize("delete", [False, True])
def test_edge_corrections_match_actual_parallel_self_and_missing_pair_mutations(delete):
	graph = dgl.graph(([0, 0, 0, 1, 2, 3], [1, 1, 1, 2, 2, 0]), num_nodes=5)
	u, v = torch.tensor([0, 0, 2, 4]), torch.tensor([1, 1, 2, 4])
	old = graph.adj_external().to_dense().T
	record = capture_edges(graph, u, v, delete=delete)
	assert graph.num_edges() == 6
	if delete:
		present = graph.has_edges_between(u, v)
		_, _, ids = graph.edge_ids(u[present], v[present], return_uv=True)
		graph.remove_edges(ids.unique())
		assert record.weights.tolist() == [-3, -1]
	else:
		graph.add_edges(u, v)
		assert record.weights.tolist() == [2, 1, 1]
	actual = graph.adj_external().to_dense().T - old
	wanted = torch.zeros_like(actual)
	wanted.index_put_((record.targets, record.sources), record.weights.to(wanted.dtype), accumulate=True)
	torch.testing.assert_close(actual, wanted)
	u.fill_(3)
	v.fill_(3)
	assert 3 not in record.sources


def test_capture_uses_sparse_old_features_for_simultaneous_structural_and_input_cross_terms():
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	changed = torch.tensor([3, 0, 0])
	snapshot = capture_features(features, changed, max_bytes=48)
	assert (
		snapshot.nodes.tolist() == [0, 3]
		and snapshot.values.untyped_storage().data_ptr() != features.untyped_storage().data_ptr()
	)
	old = graph.adj_external().to_dense().T
	old_sums = old @ features
	u, v = torch.tensor([0, 0, 3]), torch.tensor([1, 1, 3])
	edges = capture_edges(graph, u, v, delete=True)
	# Both structural sources are changed, so their exact old values are in the
	# sparse snapshot. No previous whole-graph embedding matrix is retained.
	features[snapshot.nodes] += 1.25
	_, _, ids = graph.edge_ids(u, v, return_uv=True)
	graph.remove_edges(ids.unique())
	plan = compile_delta_routes(graph, snapshot.nodes, edges.sources, edges.targets, edges.weights, max_edges=16)
	actual = update_neighbor_sums(
		old_sums[plan.targets],
		old.sum(1).to(torch.int64)[plan.targets],
		plan.inputs.bind(features[snapshot.nodes] - snapshot.values),
		plan.edges.bind(snapshot.values[torch.searchsorted(snapshot.nodes, plan.edges.value_nodes)]),
	)
	expected = graph.adj_external().to_dense().T @ features
	torch.testing.assert_close(actual.sums, expected[plan.targets], atol=2e-6, rtol=2e-6)
	torch.testing.assert_close(actual.degrees, graph.in_degrees(plan.targets))


def test_feature_capture_rejects_budget_before_copy_and_preserves_exact_old_float_bits(monkeypatch):
	features = torch.tensor([[1e20, -1e20], [3.125, 4.5], [9.0, 11.0]])
	nodes = torch.tensor([1, 0])
	original = torch.index_select
	calls = []

	def select(*args):
		calls.append(1)
		return original(*args)

	monkeypatch.setattr(torch, "index_select", select)
	with pytest.raises(PlanTooLarge, match="byte budget"):
		capture_features(features, nodes, max_bytes=31)
	assert not calls
	state = capture_features(features, nodes, max_bytes=32)
	features[:2] = 0
	assert state.values.tolist() == [[float(torch.tensor(1e20)), float(torch.tensor(-1e20))], [3.125, 4.5]]
	assert len(calls) == 1


def test_high_degree_parallel_counts_scan_bounded_csr_chunks_and_empty_capture():
	graph = dgl.graph((torch.zeros(131077, dtype=torch.int64), torch.arange(131077) % 3), num_nodes=4)
	record = capture_edges(graph, torch.tensor([0, 0, 0, 0]), torch.tensor([2, 0, 1, 2]), delete=True)
	assert record.weights.tolist() == [-43693, -43692, -43692]
	empty = torch.empty(0, dtype=torch.int64)
	assert not capture_edges(graph, empty, empty, delete=True).weights.numel()
	assert capture_features(torch.empty(0, 4), empty, max_bytes=0).values.shape == (0, 4)
