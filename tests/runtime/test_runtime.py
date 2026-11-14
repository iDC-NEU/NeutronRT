from contextlib import suppress
from threading import Event

import dgl
import pytest
import torch
import torch.multiprocessing as mp

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.data_plane import RejectedPublication
from lib.graph.ownership import MissingState
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import TransportFailure
from tests.dist.test_transport import _join_processes
from tests.runtime.test_service import _reference


def _terminate(runtime, rank):
	process = runtime.processes[rank]
	process.kill()
	process.join(5)
	assert not process.is_alive()


def _managed(_rank, scenario, use_cuda, name):
	torch.set_num_threads(1)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	pointer, edges = features.data_ptr(), graph.num_edges()
	spec = ModelSpec(name, 4, 4, 3, seed=19)
	devices = ["cuda:0"] if use_cuda else ["cpu", "cpu"]
	copy_started, copy_proceed = Event(), Event()
	with ManagedRuntime(
		graph,
		features,
		spec,
		devices,
		worker_bytes=1024 * 1024,
		storage_bytes=1024 * 1024,
		timeout=15,
		max_retries=0 if scenario == "retry_exhausted" else 2,
		batch_size=1,
		switches=RuntimeFeatures(pipeline=True),
	) as runtime:
		service, plane = runtime.service, runtime.service.remote_plane
		model = spec.build()
		query = torch.arange(6)
		service.ensure(query)
		old_owners = tuple(plane.remotes)
		old_pids = {process.pid for process in runtime.processes.values()}
		keys = tuple(group.key for group in service.catalog.selections(2, query, 0))
		producer = plane.directory.locate(keys[0]).owner
		if scenario in ("partial_update", "retry_exhausted"):
			original = plane.transport.compute
			layer_two = 0

			def interrupted(owner, packet, **kwargs):
				nonlocal layer_two
				if packet.header.layer == 2:
					layer_two += 1
					if layer_two == 2:
						assert service.counters.updates == 0
						_terminate(runtime, owner.rank)
				return original(owner, packet, **kwargs)

			plane.transport.compute = interrupted
			admission = UpdateAdmission(service)
			try:
				if scenario == "retry_exhausted":
					with pytest.raises(TransportFailure):
						admission.sync(UpdateKind.INSERT, torch.tensor([2, 4]), torch.tensor([0, 5]))
					assert runtime.reset_count == 0 and service.counters.updates == 0
					assert graph.num_edges() == edges + 2
					service.max_retries = 2
					result = service.resume_update(1)
				else:
					result = admission.sync(UpdateKind.INSERT, torch.tensor([2, 4]), torch.tensor([0, 5]))
			finally:
				admission.close()
			assert layer_two == 2 and result.version == 1
			assert service.counters.updates == 1 and service.counters.input_changes == 2
			assert graph.num_edges() == edges + 2
			assert service.resume_update(1) is result
			assert service.catalog.available(1, torch.tensor([0, 5]), 1).all()
			assert service.catalog.available(2, torch.tensor([0, 1, 5]), 1).all()
		else:
			if scenario == "backed_up":
				for key in keys:
					plane.writeback(key, plane.storage.owner).result(timeout=10)
			elif scenario == "before_ack":
				complete = plane.directory.complete_copy

				def delayed_ack(attempt, handle):
					copy_started.set()
					assert copy_proceed.wait(20)
					return complete(attempt, handle)

				plane.directory.complete_copy = delayed_ack
				pending = plane.writeback(keys[0], plane.storage.owner)
				assert copy_started.wait(10)
			before = service.counters.tasks
			_terminate(runtime, producer.rank)
			try:
				with service.query(query, plane.storage.owner) as actual:
					torch.testing.assert_close(actual, _reference(service, model), rtol=2e-5, atol=2e-5)
			finally:
				copy_proceed.set()
			if scenario == "backed_up":
				assert service.counters.tasks == before and service.counters.recovery_rows == 0
				assert tuple(group.key for group in service.catalog.selections(2, query, 0)) == keys
			else:
				assert service.counters.tasks > before and service.counters.recovery_rows > 0
				if scenario != "before_ack":
					assert plane.storage.store.bytes_used == 0
			if scenario == "before_ack":
				with pytest.raises(RejectedPublication):
					pending.result(timeout=10)
				assert plane.storage.store.bytes_used == 0
		with service.query(query, plane.storage.owner) as actual:
			torch.testing.assert_close(actual, _reference(service, model), rtol=2e-5, atol=2e-5)
		assert runtime.reset_count == 1
		assert all(owner.generation == 2 for owner in plane.remotes)
		assert not (old_pids & {process.pid for process in runtime.processes.values()})
		assert features.data_ptr() == pointer and service.graph is graph and service.features is features
		assert not torch.cuda.is_initialized()
		for key in keys:
			with suppress(MissingState):
				assert plane.directory.locate(key).owner not in old_owners
	assert not runtime.processes


@pytest.mark.parametrize("scenario", ["unflushed", "backed_up", "before_ack", "partial_update", "retry_exhausted"])
def test_master_survives_actual_worker_exit_and_recovers_owned_state(scenario):
	context = mp.spawn(_managed, args=(scenario, False, "sage"), nprocs=1, join=False)
	_join_processes(context, timeout=100)


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_gpu_process_loss_during_an_update_recomputes_before_completion(name):
	if not torch.cuda.is_available():
		pytest.skip("Requires one visible GPU")
	context = mp.spawn(_managed, args=("partial_update", True, name), nprocs=1, join=False)
	_join_processes(context, timeout=100)
