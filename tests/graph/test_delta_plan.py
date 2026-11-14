"""Sparse metadata routing agrees with full neighborhoods, including cross terms."""

import dgl
import pytest
import torch

from lib.graph.delta_plan import compile_delta_routes, select_delta_routes, slice_delta_routes
from lib.graph.deltas import update_neighbor_sums
from lib.graph.recompute import PlanTooLarge


def _graph(adjacency):
	target, source = adjacency.nonzero(as_tuple=True)
	counts = adjacency[target, source]
	return dgl.graph((source.repeat_interleave(counts), target.repeat_interleave(counts)), num_nodes=adjacency.shape[0])


def test_plan_uses_compact_changed_source_values_and_keeps_self_only_updates():
	old = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [2, 0, 1, 0], [0, 1, 0, 0]])
	new = old.clone()
	new[2, 0] = 0
	new[3, 0] = 3
	before = torch.tensor([[1.0, 5.0], [2.0, 7.0], [3.0, 11.0], [4.0, 13.0]])
	after = before.clone()
	after[[0, 3]] += 0.75
	graph = _graph(new)
	target, source = (new - old).nonzero(as_tuple=True)
	plan = compile_delta_routes(
		graph, torch.tensor([3, 0, 0]), source, target, (new - old)[target, source], max_edges=5
	)
	assert plan.inputs.value_nodes.tolist() == [0, 3]
	assert plan.inputs.sources.numel() == 3 and plan.edges.value_nodes.tolist() == [0]
	assert plan.targets.tolist() == [0, 2, 3]  # Isolated changed node 0 is still needed for fc_self.
	actual = update_neighbor_sums(
		(old.float() @ before)[plan.targets],
		old.sum(1)[plan.targets],
		plan.inputs.bind((after - before)[plan.inputs.value_nodes]),
		plan.edges.bind(before[plan.edges.value_nodes]),
	)
	torch.testing.assert_close(actual.sums, (new.float() @ after)[plan.targets])
	torch.testing.assert_close(actual.degrees, new.sum(1)[plan.targets])


def test_route_budget_rejects_before_materializing_outgoing_edges():
	class MetadataOnly:
		def num_nodes(self):
			return 3

		def out_degrees(self, nodes):
			return torch.tensor([10000000])

		def out_edges(self, nodes):
			raise AssertionError("An over-budget neighborhood must not be materialized")

	empty = torch.empty(0, dtype=torch.int64)
	with pytest.raises(PlanTooLarge, match="edge budget"):
		compile_delta_routes(MetadataOnly(), torch.tensor([1]), empty, empty, empty, max_edges=32)


def test_sparse_routes_partition_storage_reads_without_dropping_parallel_edges():
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=4)
	requests = []

	class BoundedTopology:
		response_bytes = 32

		num_nodes = graph.num_nodes
		out_degrees = graph.out_degrees

		def out_edges(self, nodes):
			assert (nodes.numel() + int(graph.out_degrees(nodes).sum())) * 8 <= self.response_bytes
			requests.append(nodes)
			return graph.out_edges(nodes)

	empty = torch.empty(0, dtype=torch.int64)
	changed = torch.arange(4)
	expected = compile_delta_routes(graph, changed, empty, empty, empty, max_edges=4)
	actual = compile_delta_routes(BoundedTopology(), changed, empty, empty, empty, max_edges=4)
	assert len(requests) > 1
	torch.testing.assert_close(actual.targets, expected.targets)
	for field in ("value_nodes", "sources", "targets"):
		torch.testing.assert_close(getattr(actual.inputs, field), getattr(expected.inputs, field))


def test_sparse_routes_reject_single_rows_exceeding_storage_capacity_before_reading():
	class OversizedRow:
		response_bytes = 32

		def num_nodes(self):
			return 1

		def out_degrees(self, nodes):
			return torch.full_like(nodes, 5)

		def out_edges(self, nodes):
			raise AssertionError("Oversized single row must fail before reading adjacency")

	empty = torch.empty(0, dtype=torch.int64)
	with pytest.raises(PlanTooLarge, match="single outgoing"):
		compile_delta_routes(OversizedRow(), torch.tensor([0]), empty, empty, empty, max_edges=5)


def test_empty_and_structure_only_plans():
	empty = torch.empty(0, dtype=torch.int64)
	graph = dgl.graph(([], []), num_nodes=4)
	plan = compile_delta_routes(graph, empty, empty, empty, empty, max_edges=0)
	assert not plan.targets.numel() and plan.inputs.value_nodes.numel() == 0
	plan = compile_delta_routes(graph, empty, torch.tensor([1]), torch.tensor([2]), torch.tensor([-3]), max_edges=1)
	assert plan.targets.tolist() == [2] and plan.edges.value_nodes.tolist() == [1]
	with pytest.raises(ValueError, match="planned source"):
		plan.edges.bind(torch.zeros(2, 4))


def test_target_slices_keep_only_needed_source_values_and_preserve_parallel_corrections():
	old = torch.tensor([[0, 0, 0, 1], [2, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
	new = old.clone()
	new[1, 0] = 0
	new[3, 2] = 3
	before = torch.arange(12, dtype=torch.float32).reshape(4, 3)
	after = before + 0.75
	target, source = (new - old).nonzero(as_tuple=True)
	plan = compile_delta_routes(_graph(new), torch.arange(4), source, target, (new - old)[target, source], max_edges=32)
	for start, stop in ((0, 1), (1, 3), (3, 4), (2, 2)):
		selected = slice_delta_routes(plan, start, stop)
		assert selected.targets.untyped_storage().data_ptr() == plan.targets.untyped_storage().data_ptr()
		assert selected.inputs.value_nodes.numel() == selected.inputs.sources.unique().numel()
		assert selected.edges.value_nodes.numel() == selected.edges.sources.unique().numel()
		actual = update_neighbor_sums(
			(old.float() @ before)[selected.targets],
			old.sum(1)[selected.targets],
			selected.inputs.bind((after - before)[selected.inputs.value_nodes]),
			selected.edges.bind(before[selected.edges.value_nodes]),
			chunk_edges=1,
		)
		torch.testing.assert_close(actual.sums, (new.float() @ after)[selected.targets])
		torch.testing.assert_close(actual.degrees, new.sum(1)[selected.targets])
	with pytest.raises(ValueError, match="interval"):
		slice_delta_routes(plan, 0, 5)


@pytest.mark.parametrize("positions", [[3, 1], [0, 2], [], [2]])
def test_arbitrary_target_selection_preserves_old_new_algebra_and_reindexes_routes(positions):
	old = torch.tensor([[0, 0, 0, 1], [2, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
	new = old.clone()
	new[1, 0] = 0
	new[3, 2] = 3
	before = torch.arange(12, dtype=torch.float32).reshape(4, 3)
	after = before + 0.75
	target, source = (new - old).nonzero(as_tuple=True)
	plan = compile_delta_routes(_graph(new), torch.arange(4), source, target, (new - old)[target, source], max_edges=32)
	selected = select_delta_routes(plan, torch.tensor(positions, dtype=torch.int64))
	assert selected.targets.tolist() == positions
	assert selected.inputs.value_nodes.numel() == selected.inputs.sources.unique().numel()
	assert selected.edges.value_nodes.numel() == selected.edges.sources.unique().numel()
	actual = update_neighbor_sums(
		(old.float() @ before)[selected.targets],
		old.sum(1)[selected.targets],
		selected.inputs.bind((after - before)[selected.inputs.value_nodes]),
		selected.edges.bind(before[selected.edges.value_nodes]),
		chunk_edges=1,
	)
	torch.testing.assert_close(actual.sums, (new.float() @ after)[selected.targets])
	torch.testing.assert_close(actual.degrees, new.sum(1)[selected.targets])


@pytest.mark.parametrize(
	"positions", [torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([4]), torch.tensor([1.0]), torch.tensor([[1]])]
)
def test_invalid_target_selection_rejects_before_routing(positions):
	empty = torch.empty(0, dtype=torch.int64)
	plan = compile_delta_routes(dgl.graph(([], []), num_nodes=4), torch.arange(4), empty, empty, empty, max_edges=0)
	with pytest.raises(ValueError, match="target positions"):
		select_delta_routes(plan, positions)
