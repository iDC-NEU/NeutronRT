"""Aggregate-only storage retains a bounded hot h cache under real pressure."""

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


@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("pipeline", [False, True])
@torch.inference_mode()
def test_tight_worker_memory_drops_h_reconstructs_inputs_and_keeps_exact_queries(
	monkeypatch, coalesced, gather, pipeline
):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		for worker in workers:
			worker.store.max_bytes = transport.endpoints[worker.owner].max_bytes = 96
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(
			graph,
			features,
			(4, 4, 3),
			plane,
			retain_aggregates=True,
			aggregate_only=True,
			batch_size=2,
			pipeline=pipeline,
			gather_inputs=gather,
		)
		writes, recipes = [], []
		copy, compute = plane._copy, transport.compute_from_aggregates

		def copy_state(attempt):
			writes.append(attempt.compute.key.kind)
			return copy(attempt)

		def recipe(*args, **kwargs):
			recipes.append(1)
			return compute(*args, **kwargs)

		monkeypatch.setattr(plane, "_copy", copy_state)
		monkeypatch.setattr(transport, "compute_from_aggregates", recipe)
		rows = torch.tensor([5, 1, 0, 1, 4, 3, 2])

		def check():
			model = workers[0].model
			expected = model(1, graph, model(0, graph, features))
			with service.query(rows, plane.storage.owner) as actual:
				torch.testing.assert_close(actual, expected[rows], atol=2e-6, rtol=2e-6)
			assert all(node.bytes_used <= 96 for node in plane.remotes.values())

		check()
		service.change_features(torch.tensor([0, 3]), torch.full((2, 4), 0.5))
		check()
		service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		check()
		service.delete_edges(torch.tensor([0, 0, 4]), torch.tensor([1, 1, 5]))
		check()
		assert recipes and writes and set(writes) == {"a"}
		assert service.counters.reconstruction_tasks > 0 and service.counters.reconstructed_rows > 0
		# Losing every cached h must not force a new neighborhood pass while sums
		# and their lower-layer dependencies remain current and readable.
		service.ensure(torch.arange(6))
		keys = {
			group.key
			for layer in (1, 2)
			for group in service.catalog.selections(
				layer,
				torch.arange(6)[service.catalog.available(layer, torch.arange(6), service.catalog.version)],
				service.catalog.version,
			)
		}
		for key in keys:
			plane.retire(key)
		before = service.counters.tasks
		check()
		assert service.counters.tasks == before
		assert features.data_ptr() == service.features.data_ptr() and not torch.cuda.is_initialized()


def test_aggregate_only_rejects_missing_sum_retention_before_startup():
	with pytest.raises(ValueError, match="retain_aggregates"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], switches=RuntimeFeatures(aggregate_only=True))
