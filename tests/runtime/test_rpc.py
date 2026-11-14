from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import dgl
import grpc
import pytest
import torch
import torch.multiprocessing as mp

from lib.graph.ownership import MissingState
from lib.models.spec import ModelSpec
from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceStub, add_OwnedInferenceServicer_to_server
from lib.rpc.limits import message_options
from lib.rpc.service import OwnedServicer, _from_wire, _to_wire
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import TransportFailure
from tests.dist.test_transport import _join_processes
from tests.graph.test_serving import _current_reference, _setup
from tests.runtime.test_runtime import _terminate
from tests.runtime.test_service import _reference


@contextmanager
def _server(service, owner, **options):
	frontend = OwnedServicer(service, owner, **options)
	with ThreadPoolExecutor(max_workers=6) as executor:
		server = grpc.server(executor, options=message_options())
		add_OwnedInferenceServicer_to_server(frontend, server)
		port = server.add_insecure_port("127.0.0.1:0")
		server.start()
		try:
			with grpc.insecure_channel(f"127.0.0.1:{port}", options=message_options()) as channel:
				yield OwnedInferenceStub(channel), frontend
		finally:
			server.stop(0).wait()
			frontend.close()


def _update(request_id, kind=wire.Update.INSERT, first=None, second=None):
	return wire.Update(
		request_id=request_id,
		kind=kind,
		first=_to_wire(torch.tensor([2]) if first is None else first),
		second=_to_wire(torch.tensor([0]) if second is None else second),
	)


@pytest.fixture
def local_rpc():
	service, plane, storage, _ = _setup("sage", batch_size=1)
	service.pipeline = True
	try:
		with _server(service, storage.owner, max_pending=1, max_tickets=4) as (stub, frontend):
			yield service, plane, stub, frontend
	finally:
		plane.close()


def test_sync_async_queries_and_completion_metadata_share_the_real_update_pipeline(local_rpc):
	service, _, stub, _ = local_rpc
	pointer = service.features.data_ptr()
	changed = service.features[[1, 5]] + 0.75
	result = stub.Sync(_update("features", wire.Update.FEATURES, torch.tensor([1, 5]), changed), timeout=10)
	assert result.ready and result.version == 1 and result.input_changes == 2
	ticket = stub.Async(_update("insert", first=torch.tensor([2, 4]), second=torch.tensor([0, 5])), timeout=5)
	assert stub.Wait(wire.WaitRequest(ticket=ticket), timeout=10).version == 2
	assert (
		stub.Sync(_update("delete", wire.Update.DELETE, torch.tensor([1]), torch.tensor([2])), timeout=10).version == 3
	)
	nodes = torch.tensor([4, 1, 1, 0, 5])
	response = stub.Query(wire.QueryRequest(nodes=_to_wire(nodes)), timeout=10)
	assert response.version == 3
	torch.testing.assert_close(_from_wire(response.values, 4096), _current_reference(service)[nodes])
	root = stub.Query(wire.QueryRequest(nodes=_to_wire(nodes), layer=0), timeout=5)
	torch.testing.assert_close(_from_wire(root.values, 4096), service.features[nodes])
	info = stub.Info(wire.Empty(), timeout=5)
	assert info.completed_updates == 3 and info.completed_input_changes == 5
	assert info.graph_version == 3 and list(info.widths) == [4, 4, 3]
	assert info.session == ticket.session and not info.HasField("incomplete_version")
	assert pointer == service.features.data_ptr()


def test_rpc_deadline_preserves_admission_and_retries_do_not_duplicate_root_changes(local_rpc):
	service, _, stub, frontend = local_rpc
	started, proceed = Event(), Event()
	apply, edges = service.apply, service.graph.num_edges()

	def blocked(*args):
		started.set()
		assert proceed.wait(10)
		return apply(*args)

	service.apply = blocked
	request = _update("deadline-update")
	try:
		with pytest.raises(grpc.RpcError) as caught:
			stub.Sync(request, timeout=0.15)
		assert caught.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
		assert started.wait(5)
		ticket = stub.Async(request, timeout=5)
		assert not stub.Wait(wire.WaitRequest(ticket=ticket, poll=True), timeout=5).ready
		for call, code in (
			(lambda: stub.Release(ticket, timeout=5), grpc.StatusCode.FAILED_PRECONDITION),
			(lambda: stub.Async(_update("queue-full"), timeout=5), grpc.StatusCode.RESOURCE_EXHAUSTED),
			(
				lambda: stub.Async(_update("deadline-update", second=torch.tensor([1])), timeout=5),
				grpc.StatusCode.INVALID_ARGUMENT,
			),
		):
			with pytest.raises(grpc.RpcError) as error:
				call()
			assert error.value.code() == code
		proceed.set()
		assert stub.Wait(wire.WaitRequest(ticket=ticket), timeout=10).version == 1
		assert stub.Sync(request, timeout=5).ticket == ticket
		assert service.graph.num_edges() == edges + 1 and frontend.admission.counters.admitted == 1
		stub.Release(ticket, timeout=5)
		with pytest.raises(grpc.RpcError) as expired:
			stub.Wait(wire.WaitRequest(ticket=ticket), timeout=5)
		assert expired.value.code() == grpc.StatusCode.NOT_FOUND
		request.session = "old-session"
		with pytest.raises(grpc.RpcError) as old_session:
			stub.Async(request, timeout=5)
		assert old_session.value.code() == grpc.StatusCode.FAILED_PRECONDITION
	finally:
		proceed.set()


def test_rpc_on_demand_query_preempts_at_a_chunk_boundary(local_rpc):
	service, _, stub, frontend = local_rpc
	started, proceed, queued = Event(), Event(), Event()
	apply, submit = service.apply, frontend.demand.submit
	observed = []

	def blocked(*args):
		started.set()
		assert proceed.wait(10)
		return apply(*args)

	def enqueue(nodes, owner, consume, **kwargs):
		def record(values):
			observed.append(service.counters.updates)
			return consume(values)

		future = submit(nodes, owner, record, **kwargs)
		queued.set()
		return future

	service.apply, frontend.demand.submit = blocked, enqueue
	try:
		ticket = stub.Async(_update("preempted"), timeout=5)
		assert started.wait(5)
		query = stub.Query.future(wire.QueryRequest(nodes=_to_wire(torch.tensor([4, 1]))), timeout=10)
		assert queued.wait(5)
		proceed.set()
		response = query.result()
		assert response.version == 1 and observed == [0]
		assert stub.Wait(wire.WaitRequest(ticket=ticket), timeout=10).ready
		torch.testing.assert_close(_from_wire(response.values, 4096), _current_reference(service)[[4, 1]])
	finally:
		proceed.set()


def test_bad_rpc_payloads_fail_before_admission_or_graph_mutation(local_rpc):
	service, _, stub, frontend = local_rpc
	for request in (_update("bad-id", first=torch.tensor([-1])), _update("bad-kind", kind=0), _update("")):
		with pytest.raises(grpc.RpcError) as caught:
			stub.Async(request, timeout=5)
		assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
	frontend.max_bytes = 8
	with pytest.raises(grpc.RpcError) as oversized:
		stub.Async(_update("too-large"), timeout=5)
	assert oversized.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
	assert frontend.admission.counters.admitted == 0 and service.catalog.version == 0


def test_missing_embedding_state_is_distinct_from_an_unknown_completion_ticket(local_rpc):
	service, _, stub, _ = local_rpc

	def unavailable(*args):
		raise MissingState("producer unavailable")

	service._load = unavailable
	with pytest.raises(grpc.RpcError) as caught:
		stub.Query(wire.QueryRequest(nodes=_to_wire(torch.tensor([0]))), timeout=10)
	assert caught.value.code() == grpc.StatusCode.UNAVAILABLE


def test_rpc_reports_incomplete_applied_versions_and_operator_resume_does_not_replay_them(local_rpc):
	service, _, stub, _ = local_rpc
	apply, edges = service.apply, service.graph.num_edges()

	def failed(*args):
		raise TransportFailure("worker unavailable")

	service.apply = failed
	with pytest.raises(grpc.RpcError) as caught:
		stub.Sync(_update("failed-update"), timeout=10)
	assert caught.value.code() == grpc.StatusCode.UNAVAILABLE
	info = stub.Info(wire.Empty(), timeout=5)
	assert info.incomplete_version == 1 and info.completed_updates == 0 and info.failed_updates == 1
	service.apply = apply
	assert service.resume_update(1).version == 1
	assert service.graph.num_edges() == edges + 1
	info = stub.Info(wire.Empty(), timeout=5)
	assert info.completed_updates == 1 and not info.HasField("incomplete_version")


def _managed_rpc(_rank, use_cuda):
	torch.set_num_threads(1)
	graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 4, 3, seed=19)
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cuda:0"] if use_cuda else ["cpu", "cpu"],
		timeout=15,
		batch_size=1,
		switches=RuntimeFeatures(pipeline=True),
	) as runtime:
		service, plane = runtime.service, runtime.service.remote_plane
		with _server(service, plane.storage.owner) as (stub, _frontend):
			nodes = _to_wire(torch.arange(6))
			stub.Query(wire.QueryRequest(nodes=nodes), timeout=20)
			compute, layer_two = plane.transport.compute, 0

			def interrupted(owner, packet, **kwargs):
				nonlocal layer_two
				if packet.header.layer == 2:
					layer_two += 1
					if layer_two == 2:
						_terminate(runtime, owner.rank)
				return compute(owner, packet, **kwargs)

			plane.transport.compute = interrupted
			result = stub.Sync(
				_update("worker-crash", first=torch.tensor([2, 4]), second=torch.tensor([0, 5])), timeout=30
			)
			assert result.version == 1 and runtime.reset_count == 1
			response = stub.Query(wire.QueryRequest(nodes=nodes), timeout=20)
			torch.testing.assert_close(
				_from_wire(response.values, 4096), _reference(service, spec.build()), rtol=2e-5, atol=2e-5
			)
			assert plane.storage.store.bytes_used == 0 and not torch.cuda.is_initialized()
			info = stub.Info(wire.Empty(), timeout=5)
			assert info.completed_updates == 1 and info.completed_input_changes == 2 and info.recovery_rows > 0


@pytest.mark.parametrize("use_cuda", [False, True])
def test_real_rpc_completion_waits_for_recovery_after_worker_process_exit(use_cuda):
	if use_cuda and not torch.cuda.is_available():
		pytest.skip("Requires one visible GPU")
	context = mp.spawn(_managed_rpc, args=(use_cuda,), nprocs=1, join=False)
	_join_processes(context, timeout=100)


def test_query_larger_than_default_grpc_limit_remains_within_admission_budget(local_rpc):
	service, _, stub, frontend = local_rpc
	# The request fits the default 4 MiB envelope; its root-feature reply does not.
	nodes = torch.arange(350000, dtype=torch.int64) % service.catalog.nodes
	response = stub.Query(wire.QueryRequest(nodes=_to_wire(nodes), layer=0), timeout=15)
	assert len(response.values.data) > 4 * 1024**2
	torch.testing.assert_close(_from_wire(response.values, 64 * 1024**2), service.features[nodes])
	frontend.demand.max_bytes = 8 * 1024**2
	with pytest.raises(grpc.RpcError) as caught:
		stub.Query(wire.QueryRequest(nodes=_to_wire(nodes), layer=0), timeout=15)
	assert caught.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
	assert frontend.demand.counters.pending_bytes == 0
