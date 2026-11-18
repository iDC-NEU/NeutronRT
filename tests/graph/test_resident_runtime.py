import ast
from types import SimpleNamespace

import dgl
import pytest
import torch

from lib.graph.resident_runtime import ResidentCapacityError, ResidentGraphSAGE, estimate_resident_bytes
from lib.models import GNNModel


def _request(kind, first, second):
	return SimpleNamespace(kind=kind, first=first, second=second)


def _oracle(model, graph, features):
	with torch.inference_mode():
		values = features
		for layer in range(model.num_layers):
			values = model(layer, graph, values)
		return values


@pytest.mark.parametrize("layers", [2, 3])
def test_resident_sage_matches_fresh_dgl_after_feature_and_topology_updates(layers):
	torch.manual_seed(71)
	graph = dgl.graph(
		(
			torch.tensor([0, 0, 0, 1, 2, 3, 4, 4]),
			torch.tensor([0, 1, 1, 2, 3, 4, 3, 3]),
		),
		num_nodes=6,
	)
	features = torch.randn(6, 4)
	model = GNNModel("sage", 4, 7, 3, layers).eval()
	runtime = ResidentGraphSAGE(graph, features, model, device="cpu", capacity_bytes=64 * 1024**2)
	runtime.bootstrap()
	oracle_graph, oracle_features = graph.clone(), features.clone()
	updates = (
		_request("features", torch.tensor([5]), torch.full((1, 4), 0.5)),
		_request("insert", torch.tensor([5, 2]), torch.tensor([2, 2])),
		_request("delete", torch.tensor([0]), torch.tensor([1])),
		_request("delete", torch.tensor([5]), torch.tensor([0])),
	)
	for request in updates:
		runtime.update(request)
		if request.kind == "features":
			oracle_features.index_copy_(0, request.first, request.second)
		elif request.kind == "insert":
			oracle_graph.add_edges(request.first, request.second)
		else:
			exists = oracle_graph.has_edges_between(request.first, request.second)
			if exists.any():
				_, _, eids = oracle_graph.edge_ids(request.first[exists], request.second[exists], return_uv=True)
				oracle_graph.remove_edges(eids.unique())
		torch.testing.assert_close(
			runtime.query(torch.arange(6)),
			_oracle(model, oracle_graph, oracle_features),
			atol=1e-6,
			rtol=1e-5,
		)
	assert runtime.graph.num_edges() == oracle_graph.num_edges()
	assert runtime.version == len(updates)


def test_resident_bootstrap_handles_empty_graph_and_isolated_feature_update():
	graph = dgl.graph(([], []), num_nodes=4)
	features = torch.eye(4, dtype=torch.float32)
	model = GNNModel("sage", 4, 4, 2, 2).eval()
	runtime = ResidentGraphSAGE(graph, features, model, device="cpu", capacity_bytes=16 * 1024**2)
	runtime.bootstrap()
	request = _request("features", torch.tensor([3]), torch.ones(1, 4))
	runtime.update(request)
	features[3] = 1
	torch.testing.assert_close(runtime.query(torch.arange(4)), _oracle(model, graph, features), atol=1e-6, rtol=1e-5)


def test_warm_updates_read_sparse_aggregates_without_full_neighbor_recompute(monkeypatch):
	from lib.graph import resident_runtime

	graph = dgl.graph(
		(torch.tensor([0, 0, 1, 2]), torch.tensor([1, 1, 2, 3])),
		num_nodes=5,
	)
	features = torch.randn(5, 4, generator=torch.Generator().manual_seed(19))
	model = GNNModel("sage", 4, 6, 3, 2).eval()
	runtime = ResidentGraphSAGE(graph, features, model, device="cpu", capacity_bytes=64 * 1024**2)
	runtime.bootstrap()
	oracle_graph, oracle_features = graph.clone(), features.clone()

	def forbidden(*args, **kwargs):
		raise AssertionError("warm resident updates must not build full-neighbor blocks")

	delta_calls = []
	original_delta = resident_runtime.delta_sage_state

	def observe_delta(*args, **kwargs):
		delta_calls.append(args[3].sums.shape)
		return original_delta(*args, **kwargs)

	monkeypatch.setattr(resident_runtime, "full_sage_state", forbidden)
	monkeypatch.setattr(resident_runtime.dgl, "in_subgraph", forbidden)
	monkeypatch.setattr(resident_runtime.dgl, "to_block", forbidden)
	monkeypatch.setattr(resident_runtime, "delta_sage_state", observe_delta)
	updates = (
		_request("features", torch.tensor([0]), torch.full((1, 4), 0.25)),
		_request("insert", torch.tensor([4, 4, 4]), torch.tensor([2, 2, 4])),
		_request("delete", torch.tensor([4]), torch.tensor([4])),
		_request("delete", torch.tensor([0]), torch.tensor([1])),
	)
	for request in updates:
		runtime.update(request)
		if request.kind == "features":
			oracle_features.index_copy_(0, request.first, request.second)
		elif request.kind == "insert":
			oracle_graph.add_edges(request.first, request.second)
		else:
			_, _, eids = oracle_graph.edge_ids(request.first, request.second, return_uv=True)
			oracle_graph.remove_edges(eids.unique())
		torch.testing.assert_close(
			runtime.query(torch.arange(5)),
			_oracle(model, oracle_graph, oracle_features),
			atol=1e-6,
			rtol=1e-5,
		)
	assert runtime.counters["full_refreshes"] == 1
	assert runtime.counters["delta_input_rows"] > 0
	assert runtime.counters["routed_messages"] > 0
	assert runtime.counters["warm_updates"] == len(updates)
	assert len(delta_calls) == len(updates) * model.num_layers


def test_resident_admission_fails_closed_before_device_copy():
	graph = dgl.graph(([0], [1]), num_nodes=2)
	features = torch.ones(2, 4)
	model = GNNModel("sage", 4, 8, 3, 2).eval()
	with pytest.raises(ResidentCapacityError, match="capacity"):
		ResidentGraphSAGE(graph, features, model, device="cpu", capacity_bytes=1)


def test_resident_admission_rejects_topology_growth_before_authoritative_mutation():
	graph = dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=3)
	features = torch.ones(3, 4)
	model = GNNModel("sage", 4, 5, 3, 2).eval()
	capacity = estimate_resident_bytes(graph, features, model, transition_edge_count=graph.num_edges())
	runtime = ResidentGraphSAGE(graph, features, model, device="cpu", capacity_bytes=capacity)
	runtime.bootstrap()
	with pytest.raises(ResidentCapacityError, match="transition"):
		runtime.update(_request("insert", torch.tensor([1]), torch.tensor([2])))
	assert runtime.graph.num_edges() == graph.num_edges()
	assert estimate_resident_bytes(graph, features, model, aggregate_dtype=torch.float64) > estimate_resident_bytes(
		graph, features, model
	)


def test_resident_runtime_has_no_managed_or_remote_execution_dependency():
	from lib.graph import resident_runtime
	from scripts.evaluate import resident as resident_replay

	for module in (resident_runtime, resident_replay):
		with open(module.__file__) as stream:
			content = stream.read()
		tree = ast.parse(content)
		imports = []
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				imports.extend(alias.name for alias in node.names)
			elif isinstance(node, ast.ImportFrom):
				imports.append(node.module or "")
		assert not any(value.startswith(("lib.runtime", "lib.rpc", "lib.dist")) for value in imports)
		assert "ManagedRuntime" not in content
		assert "torch.distributed" not in content
		assert "multiprocessing" not in content
		assert "subprocess" not in content
