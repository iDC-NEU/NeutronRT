from datetime import timedelta
from threading import Event

import dgl
import pytest
import torch
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager
from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.data_plane import RejectedPublication
from lib.graph.on_demand import OnDemand
from lib.graph.ownership import LastReplica, Owner
from lib.models import GNNModel
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.runtime.worker import OwnedWorker
from tests.dist.test_transport import _join_processes


@torch.inference_mode()
def _reference(service, model):
	values = service.features
	for layer in range(model.num_layers):
		values = model(layer, service.graph, values)
	return values


def _service(rank, rendezvous, name, use_cuda, budget, loss_before_ack=False):
	torch.set_num_threads(1)
	devices = ("cpu", "cuda:0", "cuda:1") if use_cuda else ("cpu", "cpu", "cpu")
	manager = CommManager.initialize(
		3, rank, device=devices[rank], init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	control = manager.create_epoch([0, 1, 2], timeout=timedelta(seconds=20))
	torch.manual_seed(19)
	model = GNNModel(name, 4, 4, 3, 2, aggregator_type="mean", num_heads=2).eval().to(devices[rank])
	for layer in model.layers:
		if hasattr(layer, "set_allow_zero_in_degree"):
			layer.set_allow_zero_in_degree(True)
	if rank:
		OwnedWorker(manager, model, Owner(rank, 1), max_bytes=budget, control_epoch=control).run()
		return
	transport = OwnedTransport(
		manager, control, [RemoteEndpoint(Owner(peer, 1), torch.device(devices[peer]), budget) for peer in (1, 2)]
	)
	plane = RemoteEmbeddingPlane(transport, storage_bytes=1024 * 1024, max_pending=1)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	service = DistributedEmbeddingService(
		graph, features, (4, 4, 3), plane, batch_size=2, pipeline=True, source_degree_sensitive=name == "gcn"
	)
	admission, demand = UpdateAdmission(service), OnDemand(service)
	started, proceed = Event(), Event()
	copy_started, copy_proceed = Event(), Event()
	try:
		pointer = features.data_ptr()
		query = torch.arange(6)
		service.ensure(query)
		if budget > 96:
			assert plane.storage.store.bytes_used == 0
		with service.query(query, plane.storage.owner) as actual:
			torch.testing.assert_close(actual, _reference(service, model), rtol=2e-5, atol=2e-5)
		if budget > 96:
			assert plane.storage.store.bytes_used == 0, "Selected query reads must not create full storage replicas"
		original = transport.compute
		gated = False

		def gated_compute(*args, **kwargs):
			nonlocal gated
			result = original(*args, **kwargs)
			if not gated:
				gated = True
				started.set()
				assert proceed.wait(15)
			return result

		transport.compute = gated_compute
		update = admission.submit(UpdateKind.FEATURES, torch.tensor([1, 5]), features[[1, 5]] + 0.75)
		assert started.wait(10)
		observed = []

		def consume(values):
			observed.append(service.counters.updates)
			torch.testing.assert_close(values, _reference(service, model)[[4, 1, 1, 0]], rtol=2e-5, atol=2e-5)
			return values.argmax(1).clone()

		request = demand.submit(torch.tensor([4, 1, 1, 0]), plane.storage.owner, consume)
		proceed.set()
		assert request.result(timeout=30).version == 1
		assert observed == [0]
		assert update.result(timeout=30).version == 1
		for kind, source, target in (
			(UpdateKind.INSERT, torch.tensor([2, 4]), torch.tensor([0, 5])),
			(UpdateKind.DELETE, torch.tensor([1]), torch.tensor([2])),
		):
			admission.sync(kind, source, target)
			with service.query(query, plane.storage.owner) as actual:
				torch.testing.assert_close(actual, _reference(service, model), rtol=2e-5, atol=2e-5)
		assert service.counters.updates == 3
		assert service.counters.input_changes == 5
		assert features.data_ptr() == pointer
		if budget > 96:
			assert plane.storage.store.bytes_used == 0
			key = service.catalog.selections(2, torch.tensor([0]), service.catalog.version)[0].key
			producer = plane.directory.locate(key).owner
			with pytest.raises(LastReplica):
				plane.evict(key, producer)
			if loss_before_ack:
				complete = plane.directory.complete_copy

				def delayed_ack(attempt, handle):
					copy_started.set()
					assert copy_proceed.wait(20)
					return complete(attempt, handle)

				plane.directory.complete_copy = delayed_ack
				pending = plane.writeback(key, plane.storage.owner)
				assert copy_started.wait(10)
				assert plane.directory.locate(key).owner == producer
				service.recover(producer)
				copy_proceed.set()
				with pytest.raises(RejectedPublication):
					pending.result(timeout=10)
				assert plane.storage.store.bytes_used == 0
				assert service.catalog.selections(2, torch.tensor([0]), service.catalog.version)[0].key != key
			else:
				plane.writeback(key, plane.storage.owner).result(timeout=10)
				plane.evict(key, producer)
				assert plane.directory.locate(key).owner == plane.storage.owner
				# Explicit endpoint-state loss reuses CPU replicas and recomputes only
				# missing state through the same process-owned execution path.
				assert key not in service.recover(producer)
				assert service.catalog.selections(2, torch.tensor([0]), service.catalog.version)[0].key == key
			with service.query(query, plane.storage.owner) as actual:
				torch.testing.assert_close(actual, _reference(service, model), rtol=2e-5, atol=2e-5)
		else:
			assert plane.storage.store.bytes_used > 0
			assert all(node.bytes_used <= node.max_bytes for node in transport.endpoints.values())
		assert not torch.cuda.is_initialized()
	finally:
		proceed.set()
		copy_proceed.set()
		admission.close()
		demand.close()
		plane.close()
		transport.stop()
		manager.destroy()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("use_cuda", [False, True])
def test_versioned_updates_queries_and_writeback_with_process_owned_workers(tmp_path, name, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_service, args=(str(tmp_path / "store"), name, use_cuda, 8192), nprocs=3, join=False)
	_join_processes(context, timeout=120)


@pytest.mark.parametrize("use_cuda", [False, True])
def test_resident_pressure_writes_back_safely_and_routes_storage_inputs(tmp_path, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_service, args=(str(tmp_path / "store"), "sage", use_cuda, 96), nprocs=3, join=False)
	_join_processes(context, timeout=120)


@pytest.mark.parametrize("use_cuda", [False, True])
def test_state_loss_before_writeback_ack_rejects_late_replica_and_recomputes(tmp_path, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_service, args=(str(tmp_path / "store"), "sage", use_cuda, 8192, True), nprocs=3, join=False)
	_join_processes(context, timeout=120)
