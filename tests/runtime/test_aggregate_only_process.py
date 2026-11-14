"""Tight-memory aggregate-only execution with independent storage processes."""

import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime

pytestmark = pytest.mark.host


@pytest.mark.parametrize("coalesced", [False, True])
@torch.inference_mode()
def test_three_layer_aggregate_only_runtime_keeps_queries_exact_after_eviction_and_updates(tmp_path, coalesced):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 5, 3, layers=3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=RuntimeFeatures(
			retain_aggregates=True,
			aggregate_only=True,
			pipeline=True,
			coalescing=coalesced,
			peer_fetch=False,
			gather_inputs=True,
			profiling=True,
			distributed_storage=True,
		),
		storage_partitions=3,
		storage_directory=str(tmp_path),
		worker_bytes=160,
		storage_bytes=65536,
		batch_size=2,
		timeout=30,
	) as runtime:
		service, plane = runtime.service, runtime.plane
		rows = torch.tensor([5, 1, 0, 1, 4, 3, 2])

		def check():
			current = features
			for layer in range(3):
				current = model(layer, graph, current)
			with service.query(rows, plane.storage.owner) as actual:
				torch.testing.assert_close(actual, current[rows], atol=3e-6, rtol=3e-6)
			assert all(worker.bytes_used <= 160 for worker in service.workers)
			assert not any(
				key.kind == "h"
				and plane.directory.locate(key, preferred=plane.storage.owner).owner == plane.storage.owner
				for key in plane._shapes
			)

		check()
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			service.change_features(torch.tensor([0, 3]), torch.full((2, 4), 0.5))
			check()
			report = profiler.stop()
		assert report["valid"] and report["traffic_balanced"]
		assert report["engine"]["reconstruction_tasks"] > 0
		assert sum(worker["counters"].get("tasks.aggregate_input_recipes", 0) for worker in report["workers"]) > 0
		service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		graph.add_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		check()
		service.delete_edges(torch.tensor([0, 0, 4]), torch.tensor([1, 1, 5]))
		_, _, ids = graph.edge_ids(torch.tensor([0]), torch.tensor([1]), return_uv=True)
		graph.remove_edges(ids.unique())
		check()
		assert features.data_ptr() == service.features.data_ptr() and not torch.cuda.is_initialized()
