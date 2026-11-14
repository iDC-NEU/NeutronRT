"""Host checks for bounded worker waves and failure at each protocol boundary."""

from contextlib import closing
from dataclasses import replace

import dgl
import pytest
import torch
import torch.multiprocessing as mp

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from tests.dist.test_transport import _join_processes
from tests.runtime.test_service import _reference

pytestmark = pytest.mark.host


def _runtime(_rank, name, use_cuda, mode):
	torch.set_num_threads(1)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	pointer, initial_edges = features.data_ptr(), graph.num_edges()
	spec = ModelSpec(name, 4, 4, 3)
	switches = RuntimeFeatures(concurrent_compute=True, pipeline=True, gather_inputs=True)
	if mode == "typed-proxy":
		switches = replace(switches, coalescing=False, peer_fetch=False)
	if mode == "bounded":
		switches = replace(switches, bounded_staleness=True)
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cuda:0", "cuda:1"] if use_cuda else ["cpu", "cpu"],
		worker_bytes=96 if mode == "pressure" else 65536,
		storage_bytes=65536,
		timeout=10,
		batch_size=2,
		switches=switches,
		idle_flush_seconds=60,
	) as runtime:
		service, plane = runtime.service, runtime.service.remote_plane
		nodes = torch.arange(6)
		service.ensure(nodes)
		assert service.concurrent_compute
		model = spec.build()
		transport = plane.transport
		if mode in ("prepare-crash", "launch-crash", "reply-crash", "prepare-error"):
			phase = "prepare" if mode.startswith("prepare") else "launch"
			original = getattr(transport, phase)
			calls = 0

			def interrupted(*args, **kwargs):
				nonlocal calls
				calls += 1
				if mode == "prepare-error" and calls == 2:
					raise ValueError("Injected preparation failure")
				result = original(*args, **kwargs)
				if calls == 1:
					if mode == "reply-crash":
						wait = result.decode

						def decode():
							value = wait()
							process = runtime.processes[2]
							process.kill()
							process.join(5)
							return value

						result = replace(result, decode=decode)
					elif mode != "prepare-error":
						process = runtime.processes[2]
						process.kill()
						process.join(5)
				return result

			setattr(transport, phase, interrupted)
		with closing(UpdateAdmission(service)) as admission:
			if mode == "prepare-error":
				with pytest.raises(ValueError, match="Injected"):
					admission.sync(UpdateKind.INSERT, torch.tensor([2, 4]), torch.tensor([0, 5]))
				assert runtime.reset_count == 0 and service.counters.updates == 0
				setattr(transport, phase, original)
				service.resume_update(1)
			else:
				admission.sync(UpdateKind.INSERT, torch.tensor([2, 4]), torch.tensor([0, 5]))
			assert graph.num_edges() == initial_edges + 2
			assert service.counters.updates == 1 and service.counters.input_changes == 2
			if mode.endswith("crash"):
				assert runtime.reset_count == 1
				assert all(owner.generation == 2 for owner in plane.remotes)
			for kind, first, second in (
				(UpdateKind.FEATURES, nodes, features + 0.25),
				(UpdateKind.DELETE, torch.tensor([1, 3]), torch.tensor([2, 4])),
			):
				admission.sync(kind, first, second)
				with service.query(nodes, plane.storage.owner) as values:
					torch.testing.assert_close(values, _reference(service, model), rtol=2e-5, atol=2e-5)
			assert service.counters.updates == 3 and service.counters.input_changes == 10
			if mode == "bounded":
				first = admission.submit(UpdateKind.FEATURES, nodes, torch.full_like(features, 1.75))
				second = admission.submit(UpdateKind.FEATURES, nodes, torch.full_like(features, 2.25))
				admission.sync(UpdateKind.FEATURES, nodes, torch.full_like(features, 2.75))
				assert first.result(10).input_version == 4 and second.result(10).input_version == 5
				assert service.counters.updates == 6 and service.counters.input_changes == 28
				with service.query(nodes, plane.storage.owner) as values:
					torch.testing.assert_close(values, _reference(service, model), rtol=2e-5, atol=2e-5)
		assert features.data_ptr() == pointer
		assert not torch.cuda.is_initialized()
		assert all(worker.bytes_used <= worker.max_bytes for worker in plane.remotes.values())


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize("mode", ["direct", "typed-proxy", "pressure", "bounded"])
def test_concurrent_waves_match_full_inference_with_real_workers(name, use_cuda, mode):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_runtime, args=(name, use_cuda, mode), nprocs=1, join=False)
	_join_processes(context, timeout=90)


@pytest.mark.parametrize("mode", ["prepare-crash", "launch-crash", "reply-crash", "prepare-error"])
def test_concurrent_wave_failure_remains_resumable_without_reapplying_mutation(mode):
	context = mp.spawn(_runtime, args=("sage", False, mode), nprocs=1, join=False)
	_join_processes(context, timeout=100)
