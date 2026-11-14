import pytest
import torch

from lib.profiling.storage import start, stop
from lib.storage.topology import ShardedTopology
from tests.storage.test_topology import _graph

pytestmark = pytest.mark.host


def test_storage_profiles_measure_window_deltas_and_shared_memory_without_torch(tmp_path):
	with ShardedTopology(_graph(), 3, directory=tmp_path, profiling=True) as topology:
		targets = torch.tensor([4, 0, 1, 5, 3])
		topology.compile_block(targets)
		before, samplers = start(topology)
		topology.compile_block(targets)
		report = stop(topology, before, samplers)
		assert len(report["partitions"]) == 3
		for part in report["partitions"].values():
			assert part["operations"]["rows"] == 1
			assert part["shared_response_bytes"] > 0
			assert part["seconds"]["rows"] > 0 and part["cpu_seconds"] > 0
			assert part["rss"]["peak_rss_bytes"] >= part["rss"]["start_rss_bytes"] > 0
			assert not part["torch_loaded"] and not part["cuda_initialized"]
			assert not part["response_buffers"]
		before, samplers = start(topology)
		idle = stop(topology, before, samplers)
		assert all(part["shared_response_bytes"] == 0 for part in idle["partitions"].values())


def test_disabled_topology_profiling_has_no_storage_sampler():
	before, samplers = start(None)
	assert before is None and samplers == {} and stop(None, before, samplers) is None
