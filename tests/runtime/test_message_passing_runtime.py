"""Exercise message-passing operators through the real process/P2P serving path."""

import os

import dgl
import pytest
import torch

from lib.models import MODEL_NAMES
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime


@pytest.mark.host
@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize(
	"devices", [("cpu", "cpu"), (os.environ.get("NEUTRONRT_TEST_DEVICE", "cuda:0"),)], ids=["cpu", "cuda"]
)
@torch.inference_mode()
def test_message_passing_operators_serve_streaming_updates(name, devices):
	if devices[0].startswith("cuda") and torch.cuda.device_count() <= int(devices[0].split(":")[1]):
		pytest.skip("Requested validation GPU is unavailable")
	torch.set_num_threads(1)
	graph = dgl.graph(([0, 0, 1, 2, 3, 4], [1, 1, 2, 3, 0, 1]), num_nodes=6)
	if name in ("rgcn", "rgat", "monet", "pinsage"):
		graph.edata["nrt"] = torch.tensor(
			[[i % (4 if name in ("rgcn", "rgat") else 1), 1, i / 10, -i / 20] for i in range(6)]
		)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 20
	spec = ModelSpec(name, 4, 6, 3, layers=3, heads=2)
	model = spec.build()
	with ManagedRuntime(
		graph, features, spec, devices, switches=RuntimeFeatures(operator_incremental=True), batch_size=2
	) as runtime:
		service = runtime.service
		for step in range(4):
			if step == 1:
				service.change_features(torch.tensor([0, 3]), features[[0, 3]] + 0.25)
			elif step == 2:
				service.insert_edges(
					torch.tensor([0]),
					torch.tensor([1]),
					attributes=torch.tensor([[2, 0.8, 0.1, -0.2]]) if name in ("rgcn", "rgat") else None,
				)
			elif step == 3:
				service.delete_edges(torch.tensor([0]), torch.tensor([1]))
			query = torch.tensor([5, 1, 0, 1, 3])
			service.ensure(query)
			with service.query(query, runtime.plane.storage.owner) as actual:
				expected = features
				for layer in range(3):
					expected = model(layer, graph, expected)
				torch.testing.assert_close(actual, expected[query], atol=3e-5, rtol=3e-5)


@pytest.mark.host
@torch.inference_mode()
def test_pma_topology_feeds_the_same_distributed_operator_path():
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 3, 0]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	spec = ModelSpec("gcn", 4, 5, 3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu"],
		switches=RuntimeFeatures(operator_incremental=True, pma_storage=True),
		batch_size=2,
	) as runtime:
		for step in range(3):
			if step == 1:
				u, v = torch.tensor([0]), torch.tensor([4])
				runtime.service.insert_edges(u, v)
				graph.add_edges(u, v)
			elif step == 2:
				u, v = torch.tensor([0]), torch.tensor([1])
				runtime.service.delete_edges(u, v)
				graph.remove_edges(graph.edge_ids(u, v, return_uv=True)[2])
			nodes = torch.arange(5)
			runtime.service.ensure(nodes)
			with runtime.service.query(nodes, runtime.plane.storage.owner) as actual:
				expected = model(1, graph, model(0, graph, features))
				torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.host
@torch.inference_mode()
def test_mapped_host_history_reads_on_gpu():
	from lib.graph.affected import AffectedPlanner
	from lib.models.incremental import IncrementalLayers

	device = os.environ.get("NEUTRONRT_TEST_DEVICE", "cuda:0")
	if not torch.cuda.is_available():
		pytest.skip("CUDA is unavailable")
	model = ModelSpec("gat", 4, 5, 3, layers=1).build().to(device)
	graph = dgl.graph(([0, 1, 2], [3, 3, 3]), num_nodes=4)
	x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) / 10
	task = AffectedPlanner(graph, 1).block(1, torch.tensor([3]))
	from dataclasses import replace

	device_task = replace(task, block=task.block.to(device))
	engine = IncrementalLayers(model, mapped_host_reads=True)
	engine.apply(device_task, x[task.sources])
	x[1] += 0.1
	actual = engine.apply(device_task, x[task.sources])
	expected = model(0, graph.to(device), x)[[3]]
	torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.host
@torch.inference_mode()
def test_dependency_layers_are_chunked_before_reserving_worker_outputs():
	torch.set_num_threads(1)
	nodes = torch.arange(96)
	graph = dgl.graph((torch.cat([nodes[:-1], nodes]), torch.cat([torch.full((95,), 95), nodes])), num_nodes=96)
	features = torch.arange(384, dtype=torch.float32).reshape(96, 4) / 100
	spec = ModelSpec("gat", 4, 5, 3, layers=3)
	model = spec.build()
	with ManagedRuntime(graph, features, spec, ["cpu"], worker_bytes=1024, batch_size=8, worker_threads=1) as runtime:
		runtime.service.ensure(torch.tensor([95]))
		with runtime.service.query(torch.tensor([95]), runtime.plane.storage.owner) as actual:
			expected = features
			for layer in range(3):
				expected = model(layer, graph, expected)
			torch.testing.assert_close(actual, expected[[95]], atol=2e-5, rtol=2e-5)
		assert runtime.plane.storage.store.bytes_used > 0
