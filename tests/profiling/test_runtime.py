"""Real process-group integration: queued for the consolidated host batch."""

from dataclasses import replace

import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from scripts.validate.inference import reference


@pytest.mark.parametrize(
	"flags",
	[
		{},
		{"coalescing": True},
		{"pipeline": True},
		{"peer_fetch": True},
		{"peer_fetch": True, "gather_inputs": True},
		{"gather_inputs": True},
		{"native_analysis": True},
		{"pipeline": True, "coalescing": True, "peer_fetch": True},
	],
)
@pytest.mark.parametrize("model", ["sage", "gat", "gcn"])
def test_profiled_worker_world_matches_full_inference_with_switches(flags, model, tmp_path):
	graph = dgl.graph(([0, 1, 1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 0]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec(model, 4, 5, 3)
	switches = replace(RuntimeFeatures.baseline(), profiling=True, **flags)
	with ManagedRuntime(
		graph, features, spec, ["cpu", "cpu"], switches=switches, batch_size=2, profile_dir=str(tmp_path), timeout=15
	) as runtime:
		service = runtime.service
		for layer in (1, 2):
			service.ensure(torch.arange(6), layer)
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			service.change_features(torch.tensor([1]), features[[1]] + 1)
			service.insert_edges(torch.tensor([0]), torch.tensor([4]))
			service.delete_edges(torch.tensor([1]), torch.tensor([3]))
			report = profiler.stop()
			assert report["valid"] and report["traffic_balanced"]
			assert report["engine"]["updates"] == report["engine"]["input_changes"] == 3
			assert report["engine"]["completed_changes_per_second"] > 0
			assert report["p2p"]["send_payload_bytes"] > 0
			assert len(report["workers"]) == 2
			if flags.get("gather_inputs"):
				assert sum(worker["counters"].get("gather.groups", 0) for worker in report["workers"]) > 1
			assert (
				sum(
					value
					for worker in report["workers"]
					for name, value in worker["counters"].items()
					if name.startswith("model.layer") and name.endswith(".targets")
				)
				> 0
			)
			with service.query(torch.arange(6), runtime.plane.storage.owner) as actual:
				torch.testing.assert_close(
					actual, reference(spec.build(), graph, features, torch.arange(6)), rtol=2e-5, atol=2e-5
				)
			assert not torch.cuda.is_initialized()
