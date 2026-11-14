import dgl
import pytest
import torch

from lib.graph.methods import neighborhood_policy
from lib.graph.neighborhood import split_task
from lib.graph.recompute import RecomputePlanner
from lib.models.operators import full_neighbor_forward
from lib.models.spec import ModelSpec


@pytest.fixture(autouse=True)
def sampling_threads():
	previous = torch.get_num_threads()
	torch.set_num_threads(2)
	try:
		yield
	finally:
		torch.set_num_threads(previous)


def graph():
	u = torch.arange(20).repeat_interleave(20)
	v = torch.arange(20).repeat(20)
	g = dgl.graph((u, v))
	g.edata["nrt"] = torch.stack(
		[torch.zeros(400), torch.arange(400).float() + 1, torch.zeros(400), torch.zeros(400)], 1
	)
	return g


@pytest.mark.parametrize("method,fanout", [("RTEC-Full", 20), ("RTEC-NS", 1), ("RTEC-NS", 5), ("RTEC-NS", 10)])
def test_baselines_expand_all_layers_despite_available_intermediates(method, fanout):
	g = graph()
	p = RecomputePlanner(g, 3, policy=neighborhood_policy(method, fanout=fanout, seed=23))
	plan = p.plan(
		torch.tensor([2, 4, 8]), 3, 7, available=lambda layer, nodes, version: torch.ones_like(nodes, dtype=torch.bool)
	)
	assert [t.layer for t in plan.tasks] == [1, 2, 3]
	for task in plan.tasks:
		assert (task.block.in_degrees() == fanout).all()
		u, v = task.block.edges()
		original_eids = task.sources[u] * 20 + task.targets[v]
		torch.testing.assert_close(task.block.edata["nrt"], g.edata["nrt"][original_eids])
	assert (
		not RecomputePlanner(g, 2)
		.plan(
			torch.tensor([2, 4, 8]),
			2,
			7,
			available=lambda layer, nodes, version: torch.ones_like(nodes, dtype=torch.bool),
		)
		.tasks
	)
	other = p.plan(torch.tensor([2, 4, 8]), 3, 7)
	for a, b in zip(plan.tasks, other.tasks, strict=True):
		assert torch.equal(a.sources, b.sources)
		assert torch.equal(a.block.edata["nrt"], b.block.edata["nrt"])


@pytest.mark.parametrize("model_name", ["gcn", "sage", "gat", "monet", "agnn", "gin"])
@pytest.mark.parametrize("payload_limit", [None, 5000])
@torch.inference_mode()
def test_split_sampled_tasks_preserve_native_operator_outputs_and_degrees(model_name, payload_limit):
	g = graph()
	model = ModelSpec(model_name, 4, 8, 3, layers=1).build()
	p = RecomputePlanner(
		g, 1, policy=neighborhood_policy("RTEC-NS", fanout=5, seed=17), source_degree_sensitive=model_name == "gcn"
	)
	task = p.plan(torch.arange(20), 1, 0).tasks[0]
	x = torch.randn(20, 4)
	layer = model.layers[0]
	native = layer(task.block, x[task.sources])
	outputs = [
		full_neighbor_forward(layer, chunk, x[chunk.sources])
		for chunk in split_task(task, 3, input_width=4, max_bytes=payload_limit)
	]
	actual = torch.cat(outputs)
	torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-5)


def test_task_splitting_bounds_payload_and_keeps_global_ids_out_of_graph_dimensions(monkeypatch):
	from lib.graph.neighborhood import task_payload_bytes
	from lib.graph.recompute import LayerPlan, PlanTooLarge

	block = dgl.create_block(([2, 3], [0, 1]), num_src_nodes=4, num_dst_nodes=2)
	sources = torch.tensor([10**12, 10**12 + 1, 10**12 + 2, 10**12 + 3])
	task = LayerPlan(1, block, sources, sources[:2])
	# A two-row task exceeds this limit; each single-row task fits.
	budget = task_payload_bytes(task, 128) - 1
	original = dgl.create_block

	def local_block(data, *, num_src_nodes, num_dst_nodes):
		assert num_src_nodes <= 4 and num_dst_nodes <= 2
		return original(data, num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes)

	monkeypatch.setattr(dgl, "create_block", local_block)
	chunks = list(split_task(task, 10, input_width=128, max_bytes=budget))
	assert len(chunks) == 2
	assert torch.equal(torch.cat([chunk.targets for chunk in chunks]), task.targets)
	assert all(task_payload_bytes(chunk, 128) <= budget for chunk in chunks)
	assert all(torch.equal(chunk.sources[: chunk.targets.numel()], chunk.targets) for chunk in chunks)
	with pytest.raises(PlanTooLarge, match="single destination"):
		list(split_task(task, 10, input_width=128, max_bytes=4096))
