"""Independent DGL oracles for signed updates, normalization and attention."""

import dgl
import pytest
import torch

from lib.graph.affected import AffectedPlanner
from lib.models import MODEL_NAMES
from lib.models.incremental import IncrementalLayers
from lib.models.spec import ModelSpec


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("budget", [0, 1024**2])
@torch.inference_mode()
def test_signed_stream_matches_native_full_inference(name, budget):
	torch.set_num_threads(1)
	model = ModelSpec(name, 4, 6, 3, layers=3, heads=2, seed=41).build()
	engine = IncrementalLayers(model, max_bytes=budget, refresh_steps=4)
	graph = dgl.graph(([0, 0, 1, 2, 3, 4, 5], [1, 1, 2, 3, 0, 1, 4]), num_nodes=7)
	if name in ("rgcn", "rgat", "monet", "pinsage"):
		graph.edata["nrt"] = torch.tensor(
			[[i % (4 if name in ("rgcn", "rgat") else 1), 0.5 + i / 10, i / 20, -i / 10] for i in range(7)]
		)
	x = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 15
	for step in range(15):
		if step % 3 == 0:
			x[step % 7] += 0.13
		elif step % 3 == 1:
			graph.add_edges(torch.tensor([step % 7]), torch.tensor([(step + 1) % 7]))
		else:
			graph.remove_edges(torch.tensor([0]))
		planner = AffectedPlanner(graph, 3, source_degree_sensitive=name == "gcn")
		current = x
		for layer in range(1, 4):
			# Vary ordering and chunk boundaries to exercise ID-keyed history.
			order = torch.arange(7).roll(step)
			outputs = torch.empty(7, 3 if layer == 3 else 6)
			for nodes in order.split(3):
				task = planner.block(layer, nodes)
				outputs[nodes] = engine.apply(task, current[task.sources])
			current = model(layer - 1, graph, current)
			torch.testing.assert_close(outputs, current, atol=3e-5, rtol=3e-5)
	assert engine.bytes_used <= budget
	if budget:
		assert engine.reused > 0


@pytest.mark.parametrize("name", ["pinsage", "rgcn", "rgat", "ggcn", "agnn", "monet"])
@torch.inference_mode()
def test_table_adapters_against_independent_dense_formulas(name):
	model = ModelSpec(name, 4, 6, 3, layers=1, seed=9).build()
	layer = model.layers[0]
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 1, 0]), num_nodes=5)
	x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	attrs = torch.tensor([[i % (4 if name in ("rgcn", "rgat") else 1), 0.7, i / 10, -i / 20] for i in range(5)])
	graph.edata["nrt"] = attrs
	u, v = graph.edges()
	expected = torch.zeros(5, 3)
	for target in range(5):
		for relation in range(layer.relations):
			selected = (v == target) & (attrs[:, 0] == relation)
			sources = x[u[selected]]
			if not sources.numel():
				continue
			if name == "pinsage":
				projected = torch.relu(sources @ layer.project.weight.T + layer.project.bias)
				value = ((projected @ layer.fc.weight.T) * attrs[selected, 1:2]).mean(0)
			elif name == "rgcn":
				value = (sources @ layer.rel_weight[relation]).mean(0)
			elif name == "rgat":
				projected = sources @ layer.rel_weight[relation]
				destination = x[target] @ layer.rel_weight[relation]
				score = torch.nn.functional.leaky_relu(
					projected @ layer.attn_l[relation] + destination @ layer.attn_r[relation], 0.2
				)
				value = (score.softmax(0).unsqueeze(1) * projected).sum(0)
			elif name == "ggcn":
				gate = torch.sigmoid(
					sources @ layer.source_gate.weight.T
					+ layer.source_gate.bias
					+ x[target] @ layer.destination_gate.weight.T
				)
				value = ((sources * gate) @ layer.fc.weight.T).sum(0)
			elif name == "agnn":
				cosine = torch.nn.functional.cosine_similarity(sources, x[target].expand_as(sources), dim=1)
				value = ((sources @ layer.fc.weight.T) * (layer.beta * cosine).softmax(0).unsqueeze(1)).sum(0)
			else:
				# Independent native DGL Gaussian mixture convolution.
				conv = dgl.nn.GMMConv(4, 3, 2, 3, bias=False, allow_zero_in_degree=True)
				conv.fc.weight.copy_(layer.kernel_weight.permute(0, 2, 1).reshape(9, 4))
				conv.mu.copy_(layer.mu)
				conv.inv_sigma.copy_(layer.inv_sigma)
				value = conv(graph, x, attrs[:, 2:])[target]
			expected[target] += value
		if name in ("pinsage", "rgcn"):
			expected[target] += x[target] @ layer.fc_self.weight.T + layer.fc_self.bias
	engine = IncrementalLayers(model)
	task = AffectedPlanner(graph, 1).block(1, torch.arange(5))
	torch.testing.assert_close(engine.apply(task, x[task.sources]), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("name", ["gcn", "gat", "rgat", "sage"])
@torch.inference_mode()
def test_only_changed_messages_are_recomputed_and_attention_refreshes_destination(name):
	model = ModelSpec(name, 4, 4, 3, layers=1).build()
	graph = dgl.graph((torch.arange(100), torch.full((100,), 100)), num_nodes=101)
	x = torch.arange(404, dtype=torch.float32).reshape(101, 4) / 1000
	task = AffectedPlanner(graph, 1, source_degree_sensitive=name == "gcn").block(1, torch.tensor([100]))
	engine = IncrementalLayers(model)
	engine.apply(task, x[task.sources])
	before = engine.messages
	x[0] += 0.1
	actual = engine.apply(task, x[task.sources])
	assert engine.messages - before == 2
	torch.testing.assert_close(actual, model(0, graph, x)[[100]], atol=2e-6, rtol=2e-6)
	if name in ("gat", "rgat"):
		before = engine.refreshes
		x[100] += 20
		actual = engine.apply(task, x[task.sources])
		assert engine.refreshes > before
		torch.testing.assert_close(actual, model(0, graph, x)[[100]], atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_agnn_attention_matches_native_dgl_before_the_output_projection():
	model = ModelSpec("agnn", 4, 5, 3, layers=1).build()
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 1]), num_nodes=4)
	features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
	layer = model.layers[0]
	native = dgl.nn.AGNNConv(init_beta=float(layer.beta), allow_zero_in_degree=True).eval()
	expected = layer.fc(native(graph, features))
	torch.testing.assert_close(model(0, graph, features), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("name", ["gat", "rgat", "agnn"])
@torch.inference_mode()
def test_attention_corrections_rebase_when_all_source_messages_become_zero(name):
	model = ModelSpec(name, 4, 7, 3, layers=1, seed=41).build()
	generator = torch.Generator().manual_seed(17)
	graph = dgl.graph((torch.cat([torch.arange(30), torch.tensor([0, 0])]), torch.full((32,), 30)), num_nodes=31)
	features = torch.randn(31, 4, generator=generator)
	task = AffectedPlanner(graph, 1).block(1, torch.tensor([30]))
	engine = IncrementalLayers(model)
	engine.apply(task, features[task.sources])
	for source in range(5):
		features[source] += torch.randn(4, generator=generator) * 0.1
		engine.apply(task, features[task.sources])
	features[:30].zero_()
	actual = engine.apply(task, features[task.sources])
	expected = model(0, graph, features)[[30]]
	assert torch.equal(expected, torch.zeros_like(expected))
	assert torch.equal(actual, expected)
