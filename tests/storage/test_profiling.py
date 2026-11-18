from threading import Event
from time import monotonic
from types import SimpleNamespace

import pytest
import torch

from lib.profiling.storage import RemoteRSSSampler, start, stop
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


def test_remote_storage_reports_sampled_peak_separately_from_endpoint_stats():
	ready = Event()

	class RemoteStorage:
		remote = True

		def __init__(self):
			self.calls = 0

		def stats(self):
			return {
				0: {
					"rss_bytes": 100,
					"pss_bytes": 80,
					"uss_bytes": 60,
					"overlay_bytes": 40,
					"overlay_metadata_bytes": 8,
					"overlay_rows": 2,
					"cpu_seconds": 1.0,
					"shared_response_bytes": 0,
					"operations": {},
					"seconds": {},
					"torch_loaded": False,
					"cuda_initialized": False,
					"response_buffers": 0,
				}
			}

		def sample_rss(self, *, timeout=None):
			self.calls += 1
			ready.set()
			return {0: {"rss_bytes": 40 + self.calls}}

	storage = RemoteStorage()
	topology = SimpleNamespace(storage=storage, initialization={})
	before, samplers = start(topology, interval=0.001)
	assert ready.wait(2)
	report = stop(topology, before, samplers)
	part = report["partitions"][0]
	assert part["rss"]["sampled_peak_rss_bytes"] >= 41
	assert part["rss"]["peak_rss_bytes"] == part["rss"]["sampled_peak_rss_bytes"]
	assert part["rss"]["end_rss_bytes"] >= part["rss"]["start_rss_bytes"]
	assert part["rss"]["samples"] >= 1
	assert part["memory_end"]["rss_bytes"] == 100
	assert report["sampling_complete"]


def test_remote_rss_sampler_stop_is_bounded_when_reader_blocks():
	entered, release = Event(), Event()

	class BlockedStorage:
		def sample_rss(self, *, timeout=None):
			entered.set()
			assert release.wait(2)
			return {0: {"rss_bytes": 10}}

	sampler = RemoteRSSSampler(BlockedStorage(), interval=0.5)
	sampler.start()
	assert entered.wait(2)
	started = monotonic()
	report = sampler.stop(timeout=0.05)
	assert monotonic() - started < 0.5
	assert not sampler.stopped
	assert report == {}
	release.set()
	sampler._thread.join(2)
	assert not sampler._thread.is_alive()
