"""Temporary recomputation inputs survive their consumers, then release ownership."""

import dgl
import pytest
import torch

from lib.graph.methods import neighborhood_policy
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime

pytestmark = pytest.mark.host


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("method", ["RTEC-Full", "RTEC-NS"])
def test_remote_plans_retain_only_final_outputs_after_dependencies_finish(method, concurrent):
	nodes = torch.arange(24)
	graph = dgl.graph((nodes.repeat_interleave(3), (nodes.repeat_interleave(3) + torch.arange(72) % 3) % 24))
	features = torch.arange(96, dtype=torch.float32).reshape(24, 4) / 100
	spec = ModelSpec("sage", 4, 5, 3, layers=3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"] if concurrent else ["cpu"],
		worker_threads=1,
		batch_size=4,
		worker_bytes=65536,
		switches=RuntimeFeatures(concurrent_compute=concurrent, profiling=True),
		neighborhood_policy=neighborhood_policy(method, fanout=5),
	) as runtime:
		service = runtime.service
		if concurrent:
			# Force dependency blocks to split into genuine two-worker waves.
			service.planner.batch_size = 8
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			service.ensure(nodes)
			report = profiler.stop()
		final_bytes = nodes.numel() * 3 * 4
		assert sum(owner["h_bytes"] for owner in report["retained_state"]["owners"]) == final_bytes
		if not concurrent:
			# Peer transfers can retain retired buffers until their read leases drain.
			assert sum(worker["resident_current_bytes"] for worker in report["workers"]) == final_bytes
		for step in range(3):
			if step == 1:
				service.change_features(torch.tensor([0]), features[[0]] + 0.5)
			elif step == 2:
				service.insert_edges(torch.tensor([0]), torch.tensor([1]))
			with torch.inference_mode():
				expected = features
				for layer in range(3):
					expected = model(layer, graph, expected)
			with service.query(nodes, runtime.plane.storage.owner) as actual:
				torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
			assert not service.catalog.available(1, nodes, service.catalog.version).any()
			assert not service.catalog.available(2, nodes, service.catalog.version).any()
