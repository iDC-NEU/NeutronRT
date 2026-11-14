import json
from datetime import timedelta
from pathlib import Path
from threading import RLock
from unittest.mock import Mock

import dgl
import pytest
import torch
import torch.multiprocessing as mp

from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.profiling.communication import ProfiledCommManager
from lib.profiling.control import _exchange, start_workers, stop_workers
from lib.profiling.worker import ProfiledWorker
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import PROFILE
from lib.runtime.service import DistributedEmbeddingService
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from tests.dist.test_transport import _join_processes
from tests.runtime.test_service import _reference


def _payload_counters(report, direction):
	return sum(
		value
		for name, value in report["counters"].items()
		if name.startswith(f"p2p.{direction}.") and name.endswith(".payload_bytes")
	)


def _profile_worker(rank, rendezvous, folder, use_cuda, trace):
	torch.set_num_threads(1)
	device = "cuda:0" if rank and use_cuda else "cpu"
	manager = ProfiledCommManager.initialize(
		2, rank, device=device, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	control = manager.create_epoch([0, 1], timeout=timedelta(seconds=20))
	spec = ModelSpec("sage", 4, 4, 3, seed=19)
	if rank:
		ProfiledWorker(
			manager,
			spec.build().to(device),
			Owner(rank, 1),
			max_bytes=1024**2,
			control_epoch=control,
			profile_dir=folder,
		).run()
		return
	transport = OwnedTransport(
		manager,
		control,
		[RemoteEndpoint(Owner(1, 1), torch.device("cuda:0" if use_cuda else "cpu"), 1024**2)],
	)
	plane = RemoteEmbeddingPlane(transport, storage_bytes=1024**2)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	service = DistributedEmbeddingService(graph, features, (4, 4, 3), plane, pipeline=True)
	try:
		service.ensure(torch.arange(6))
		start_workers(transport, trace=trace, trace_task_limit=1)
		manager.recorder.start()
		service.change_features(torch.tensor([1]), features[[1]] + 0.5)
		root_report = manager.recorder.stop()
		report = stop_workers(transport)[0]
		assert report["rank"] == report["generation"] == 1
		assert report["open_spans"] == 0 and not report["series_dropped"]
		assert report["counters"]["model.layer1.executions"] == 1
		assert report["counters"]["model.layer2.executions"] == 1
		assert report["counters"]["model.layer1.targets"] == 2
		assert report["counters"]["model.layer2.targets"] == 3
		assert report["counters"]["model.layer1.compiled_edges"] == 3
		assert report["counters"]["model.layer2.compiled_edges"] == 5
		assert report["resident_peak_bytes"] > 0 and report["rss"]["peak_rss_bytes"] > 0
		assert _payload_counters(root_report, "send") == _payload_counters(report, "recv")
		assert _payload_counters(root_report, "recv") == _payload_counters(report, "send")
		if use_cuda:
			assert report["timings"]["worker.model_cuda.layer1"]["total_seconds"] > 0
			assert report["cuda"]["peak_allocated_bytes"] >= report["resident_peak_bytes"]
		else:
			assert report["cuda"] is None and "worker.model_cuda.layer1" not in report["timings"]
		if trace:
			assert report["trace_capped"]
			assert json.loads(Path(report["torch_trace"]).read_text())["traceEvents"]
			assert json.loads(Path(report["trace_path"]).read_text())["traceEvents"]
		start_workers(transport)
		with service.query(torch.arange(6), plane.storage.owner) as actual:
			torch.testing.assert_close(actual, _reference(service, spec.build()), rtol=2e-5, atol=2e-5)
		second = stop_workers(transport)[0]
		assert not any(name.startswith("model.") for name in second["counters"])
		assert second["trace_path"] is None and second["torch_trace"] is None
		assert not torch.cuda.is_initialized()
	finally:
		plane.close()
		transport.stop()
		manager.destroy()


@pytest.mark.parametrize("use_cuda,trace", [(False, False), (False, True), (True, False), (True, True)])
def test_worker_windows_count_real_work_cuda_events_and_payloads_without_warmup(tmp_path, use_cuda, trace):
	if use_cuda and not torch.cuda.is_available():
		pytest.skip("Requires one visible GPU")
	context = mp.spawn(
		_profile_worker, args=(str(tmp_path / "store"), str(tmp_path / "traces"), use_cuda, trace), nprocs=2, join=False
	)
	_join_processes(context, timeout=80)


@pytest.mark.parametrize("generation,size", [(1, 100), (2, 8 * 1024**2 + 1)])
def test_old_generation_or_oversized_reports_are_rejected_before_data_allocation(generation, size):
	transport = Mock()
	transport.locked.return_value = RLock()
	transport.comm.recv_tensor.return_value = torch.tensor([PROFILE, 2, generation, size])
	with pytest.raises(RuntimeError):
		_exchange(transport, Owner(1, 2), 2, False, 0)
	assert transport.comm.recv_tensor.call_count == 1
