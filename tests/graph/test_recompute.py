from types import SimpleNamespace
from unittest.mock import Mock

import dgl
import dgl.nn as dglnn
import pytest
import torch
from torch import nn

from lib.graph.recompute import PlanTooLarge, RecomputePlanner, execute_recomputation


@torch.inference_mode()
def test_dependency_planning_chunks_intermediate_rows_and_retries_storage_response_limits():
	from lib.graph.recompute import compile_block

	graph = _graph()
	model = _model("sage", 2)
	features = torch.randn(6, 4)
	reference = features
	for index, layer in enumerate(model):
		reference = _apply(layer, graph, reference, index == 1)

	def bounded_block(nodes):
		if nodes.numel() > 2:
			raise MemoryError("Selected adjacency exceeds the shared response budget")
		return compile_block(graph, nodes)

	backend = SimpleNamespace(
		device=graph.device, num_nodes=graph.num_nodes, in_degrees=graph.in_degrees, compile_block=bounded_block
	)
	requested = torch.tensor([0, 3, 4])
	plan = RecomputePlanner(backend, 2, batch_size=3).plan(requested, 2, 0)
	assert len(plan.tasks) > 2 and all(task.targets.numel() <= 2 for task in plan.tasks)
	tables = [features, torch.zeros(6, 4), torch.zeros(6, 3)]
	execute_recomputation(
		plan,
		load=lambda layer, nodes, version: tables[layer][nodes],
		apply=lambda task, values: _apply(model[task.layer - 1], task.block, values, task.layer == 2),
		publish=lambda layer, nodes, values, version: tables[layer].index_copy_(0, nodes, values),
	)
	torch.testing.assert_close(tables[2][requested], reference[requested])


def _graph():
	# Parallel edges, a self-loop, a cycle, and an isolated destination.
	return dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)


def _model(name, layers):
	torch.manual_seed(42)
	modules = nn.ModuleList()
	for layer in range(layers):
		width = 3 if layer + 1 == layers else 4
		if name == "sage":
			modules.append(dglnn.SAGEConv(4, width, "mean"))
		else:
			modules.append(dglnn.GATConv(4, width, 2, residual=True, allow_zero_in_degree=True))
	return modules.eval()


def _apply(module, block, features, final):
	output = module(block, features)
	if output.ndim == 3:
		output = output.mean(1)
	return output if final else output.relu()


@torch.inference_mode()
def _check_recomputation(name, layers, device="cpu", reuse=False):
	graph = _graph()
	model = _model(name, layers).to(device)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(device) / 10
	full_graph = graph.to(device)
	reference = [features]
	for index, module in enumerate(model):
		reference.append(_apply(module, full_graph, reference[-1], index + 1 == layers))
	tables = [features, *[torch.zeros_like(value) for value in reference[1:]]]
	valid = [torch.ones(6, dtype=torch.bool), *[torch.zeros(6, dtype=torch.bool) for _ in model]]
	if reuse:
		valid[1][[0, 1, 5]] = True
		tables[1][[0, 1, 5]] = reference[1][[0, 1, 5]]
	versions = []

	def available(layer, nodes, version):
		versions.append(version)
		return valid[layer][nodes]

	def load(layer, nodes, version):
		versions.append(version)
		assert valid[layer][nodes].all()
		return tables[layer].index_select(0, nodes.to(device))

	def apply(task, inputs):
		assert inputs.device == torch.device(device)
		return _apply(model[task.layer - 1], task.block.to(device), inputs, task.layer == layers)

	def publish(layer, nodes, values, version):
		versions.append(version)
		assert values.device == torch.device(device)
		tables[layer].index_copy_(0, nodes.to(device), values)
		valid[layer][nodes] = True

	query = torch.tensor([5, 3, 3, 0])
	planner = RecomputePlanner(graph, layers)
	plan = planner.plan(query, layers, 8, available=available)
	assert planner.graph is graph
	assert plan.requested.data_ptr() == query.data_ptr()
	assert [task.layer for task in plan.tasks] == sorted(task.layer for task in plan.tasks)
	if reuse:
		assert not set(plan.tasks[0].targets.tolist()) & {0, 1, 5}
	execute_recomputation(plan, load=load, apply=apply, publish=publish)
	torch.testing.assert_close(tables[-1][query], reference[-1][query], rtol=2e-5, atol=2e-5)
	assert versions and set(versions) == {8}
	assert graph.num_edges() == 7


@pytest.mark.parametrize("model", ["sage", "gat"])
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_full_neighbor_recomputation_matches_full_inference(model, layers):
	_check_recomputation(model, layers)


@pytest.mark.parametrize("model", ["sage", "gat"])
def test_recomputation_reuses_valid_intermediate_inputs(model):
	_check_recomputation(model, 3, reuse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a visible GPU")
@pytest.mark.parametrize("model", ["sage", "gat"])
def test_real_gpu_operators_match_full_inference(model):
	_check_recomputation(model, 3, device="cuda:0", reuse=True)


def test_available_outputs_and_empty_queries_need_no_compute():
	planner = RecomputePlanner(_graph(), 2)
	apply = Mock()
	for nodes in (torch.empty(0, dtype=torch.int64), torch.tensor([1, 4])):
		plan = planner.plan(nodes, 2, 1, available=lambda layer, ids, version: torch.ones_like(ids, dtype=torch.bool))
		assert not plan.tasks
		execute_recomputation(plan, load=Mock(), apply=apply, publish=Mock())
	apply.assert_not_called()


@pytest.mark.parametrize("nodes", [torch.tensor([-1]), torch.tensor([6]), torch.ones(2), torch.ones(1, 2)])
def test_planner_rejects_invalid_node_ids(nodes):
	with pytest.raises(ValueError, match="node ID"):
		RecomputePlanner(_graph(), 2).plan(nodes, 2, 0)


def test_planner_rejects_missing_features_invalid_availability_and_oversized_plans(monkeypatch):
	graph = _graph()
	planner = RecomputePlanner(graph, 2)
	with pytest.raises(ValueError, match="Availability"):
		planner.plan(torch.tensor([1]), 1, 0, available=lambda *_: torch.ones(1))
	with pytest.raises(KeyError, match="features"):
		planner.plan(
			torch.tensor([1]), 1, 0, available=lambda layer, nodes, version: torch.zeros_like(nodes, dtype=torch.bool)
		)
	with pytest.raises(ValueError, match="layer"):
		planner.plan(torch.tensor([1]), 3, 0)
	with pytest.raises(ValueError, match="layer"):
		planner.plan(torch.tensor([1]), 1, -1)
	construct = Mock(side_effect=AssertionError("Must check budget before allocating topology"))
	monkeypatch.setattr(dgl, "in_subgraph", construct)
	with pytest.raises(PlanTooLarge):
		RecomputePlanner(graph, 2, max_edges=0).plan(torch.tensor([1]), 2, 0)
	construct.assert_not_called()


def test_streamed_plans_bound_complete_dependencies_and_recheck_published_rows():
	graph = _graph()
	planner = RecomputePlanner(graph, 2, max_edges=7)
	valid = torch.zeros((3, graph.num_nodes()), dtype=torch.bool)
	valid[0] = True
	requested = torch.tensor([5, 3, 2, 4, 1, 0, 3])
	plans = planner.plans(requested, 2, 0, available=lambda layer, nodes, version: valid[layer, nodes])
	first = next(plans)
	assert sum(task.block.num_edges() for task in first.tasks) <= 7
	# Simulate a consumer publishing rows before advancing the streaming planner.
	valid[1:] = True
	remaining = list(plans)
	assert remaining and all(not plan.tasks for plan in remaining)
	assert set(torch.cat([first.targets, *[plan.targets for plan in remaining]]).tolist()) == set(requested.tolist())


def test_streamed_plans_keep_the_single_destination_limit():
	planner = RecomputePlanner(_graph(), 2, max_edges=0)
	with pytest.raises(PlanTooLarge, match="edge budget"):
		list(planner.plans(torch.tensor([3, 3]), 2, 0))


def test_executor_validates_payload_shapes_before_publishing():
	plan = RecomputePlanner(_graph(), 1).plan(torch.tensor([1, 3]), 1, 0)
	publish = Mock()
	with pytest.raises(ValueError, match="Input rows"):
		execute_recomputation(plan, load=lambda *_: torch.zeros(0, 4), apply=Mock(), publish=publish)
	with pytest.raises(ValueError, match="Output rows"):
		execute_recomputation(
			plan,
			load=lambda layer, nodes, version: torch.zeros(nodes.numel(), 4),
			apply=lambda *_: torch.zeros(0, 3),
			publish=publish,
		)
	publish.assert_not_called()
