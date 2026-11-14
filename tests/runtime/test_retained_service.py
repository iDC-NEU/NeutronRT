from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.managed import ManagedRuntime
from lib.runtime.service import DistributedEmbeddingService
from tests.runtime.test_prepared import _actors


@torch.inference_mode()
def _check_retained(service, model, graph, features):
	nodes = torch.arange(graph.num_nodes())
	service.ensure(nodes)
	current = features
	for index in range(2):
		projected = model.layers[index].fc_neigh(current)
		expected_sums = torch.zeros_like(projected)
		u, v = graph.edges()
		expected_sums.index_add_(0, v, projected[u])
		current = model(index, graph, current)
		service.ensure(nodes, index + 1)
		catalog = service.aggregate_catalog
		assert catalog.version == service.catalog.version and catalog.available(index + 1, nodes, catalog.version).all()
		for group in catalog.selections(index + 1, nodes, catalog.version):
			with ExitStack() as leases:
				actual = service.remote_plane.read_rows(group.key, group.rows, leases)
				torch.testing.assert_close(actual, expected_sums[nodes[group.positions]], atol=2e-6, rtol=2e-6)
		with service.query(nodes, service.remote_plane.storage.owner, layer=index + 1) as actual:
			torch.testing.assert_close(actual, current, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("pipeline", [False, True])
def test_retained_sums_follow_features_parallel_edges_and_empty_updates(coalesced, pipeline):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=True))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(
			graph,
			features,
			(4, 4, 3),
			plane,
			retain_aggregates=True,
			batch_size=2,
			pipeline=pipeline,
			gather_inputs=True,
		)
		_check_retained(service, workers[0].model, graph, features)
		service.change_features(torch.tensor([0, 3]), torch.full((2, 4), 0.5))
		_check_retained(service, workers[0].model, graph, features)
		service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		_check_retained(service, workers[0].model, graph, features)
		service.delete_edges(torch.tensor([0, 0, 4]), torch.tensor([1, 1, 5]))
		_check_retained(service, workers[0].model, graph, features)
		before = service.counters.tasks
		service.change_features(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
		_check_retained(service, workers[0].model, graph, features)
		assert service.counters.tasks == before and service.counters.updates == 4
		assert service.catalog.version == service.aggregate_catalog.version == 4
		assert not service._pending_analysis()


def test_partial_sum_publication_resumes_without_replaying_root_mutation(monkeypatch):
	graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=4)
	features = torch.zeros(4, 4)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(graph, features, (4, 4, 3), plane, retain_aggregates=True, batch_size=2)
		_check_retained(service, workers[0].model, graph, features)
		original = service.aggregate_catalog.publish
		failed = False

		def fail_once(key):
			nonlocal failed
			if not failed:
				failed = True
				raise ValueError("sum publication")
			return original(key)

		monkeypatch.setattr(service.aggregate_catalog, "publish", fail_once)
		with pytest.raises(ValueError, match="sum publication"):
			service.change_features(torch.tensor([0]), torch.ones(1, 4))
		assert service.counters.updates == 0 and service.catalog.version == 1
		service.resume_update(1)
		_check_retained(service, workers[0].model, graph, features)
		assert service.counters.updates == 1 and torch.equal(features[0], torch.ones(4))
		assert service.catalog.version == service.aggregate_catalog.version == 1


def test_missing_sum_is_recomputed_even_when_its_embedding_remains_readable():
	graph = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=4)
	features = torch.ones(4, 4)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(graph, features, (4, 4, 3), plane, retain_aggregates=True, batch_size=2)
		_check_retained(service, workers[0].model, graph, features)
		nodes = torch.arange(4)
		key = service.aggregate_catalog.selections(2, nodes, 0)[0].key
		plane.retire(key)
		assert service.catalog.available(2, nodes, 0).all()
		assert not service.aggregate_catalog.available(2, nodes, 0).all()
		before = service.counters.tasks
		_check_retained(service, workers[0].model, graph, features)
		assert service.counters.tasks > before and service.aggregate_catalog.available(2, nodes, 0).all()


@pytest.mark.parametrize("name", ["gat", "gcn"])
def test_invalid_model_rejected_before_worker_start(name, monkeypatch):
	graph = dgl.graph(([], []), num_nodes=2)
	monkeypatch.setattr(ManagedRuntime, "_start", lambda *_args: pytest.fail("Worker started before validation"))
	with pytest.raises(ValueError, match="mean GraphSAGE"):
		ManagedRuntime(
			graph,
			torch.zeros(2, 4),
			ModelSpec(name, 4, 4, 3),
			["cpu"],
			switches=RuntimeFeatures(retain_aggregates=True),
		)


@pytest.mark.parametrize("extra", [{"bounded_staleness": True}, {"concurrent_compute": True}])
def test_unintegrated_modes_rejected_before_worker_start(extra, monkeypatch):
	graph = dgl.graph(([], []), num_nodes=2)
	monkeypatch.setattr(ManagedRuntime, "_start", lambda *_args: pytest.fail("Worker started before validation"))
	with pytest.raises(NotImplementedError, match="serial exact dispatch"):
		ManagedRuntime(
			graph,
			torch.zeros(2, 4),
			ModelSpec("sage", 4, 4, 3),
			["cpu"],
			switches=RuntimeFeatures(retain_aggregates=True, **extra),
		)
