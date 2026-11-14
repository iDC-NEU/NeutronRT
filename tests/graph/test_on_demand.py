from threading import Event

import pytest
import torch

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.on_demand import OnDemand
from tests.graph.test_serving import _current_reference, _setup


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize(
	"devices",
	[
		("cpu", "cpu"),
		pytest.param(
			("cuda:0", "cuda:1"),
			marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs"),
		),
	],
)
@pytest.mark.parametrize("preemption", [False, True])
def test_query_preempts_update_at_a_ready_chunk_and_matches_current_version(name, devices, preemption):
	service, plane, storage, _ = _setup(name, devices=devices, batch_size=1)
	service.ensure(torch.arange(6))
	service.pipeline = True
	service.preemption_enabled = preemption
	admission, demand = UpdateAdmission(service), OnDemand(service)
	started, proceed = Event(), Event()
	apply = service.apply
	blocked = False

	def gated_apply(worker, task, inputs):
		nonlocal blocked
		if not blocked:
			blocked = True
			started.set()
			assert proceed.wait(10)
		return apply(worker, task, inputs)

	service.apply = gated_apply
	observed = []
	nodes = torch.tensor([4, 1, 1, 0])

	def consume(values):
		observed.append(service.counters.updates)
		torch.testing.assert_close(values.cpu(), _current_reference(service)[nodes], rtol=2e-5, atol=2e-5)
		return values.argmax(1).cpu().clone()

	try:
		update = admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1)
		assert started.wait(5)
		query = demand.submit(nodes, storage.owner, consume)
		assert not query.done()
		proceed.set()
		reply = query.result(timeout=20)
		assert reply.version == 1
		assert observed == [0 if preemption else 1]
		assert update.result(timeout=20).version == 1
		with service.query(torch.arange(6), storage.owner) as values:
			torch.testing.assert_close(values, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		proceed.set()
		admission.close()
		demand.close()
		plane.close()
	assert demand.counters.completed == 1
	assert demand.counters.pending == demand.counters.pending_bytes == 0


def test_idle_queries_recompute_missing_state_and_snapshot_only_requested_ids():
	service, plane, storage, _ = _setup("gat", batch_size=2)
	demand = OnDemand(service)
	try:
		nodes = torch.tensor([5, 2, 2])

		def consume(values):
			return values.cpu().clone()

		reply = demand.submit(nodes, storage.owner, consume).result(timeout=10)
		assert reply.version == 0
		torch.testing.assert_close(reply.value, _current_reference(service)[nodes])
		assert demand.counters.recomputed_rows > 0
	finally:
		demand.close()
		plane.close()
	assert service.checkpoint is None


def test_query_capacity_and_consumer_failure_do_not_corrupt_update_state():
	service, plane, storage, _ = _setup("sage")
	demand = OnDemand(service, max_pending=1, max_bytes=4096)
	started, proceed = Event(), Event()

	def fail(_values):
		started.set()
		assert proceed.wait(10)
		raise RuntimeError("response serialization failed")

	try:
		future = demand.submit(torch.tensor([1]), storage.owner, fail)
		assert started.wait(5)
		with pytest.raises(AdmissionFull):
			demand.submit(torch.tensor([1]), storage.owner, lambda values: None)
		assert not future.cancel()
		proceed.set()
		with pytest.raises(RuntimeError, match="serialization failed"):
			future.result(timeout=10)
		assert service.catalog.version == 0
		assert service.counters.updates == 0
	finally:
		proceed.set()
		demand.close()
		plane.close()
	assert demand.counters.failed == 1
	with pytest.raises(RuntimeError, match="closed"):
		demand.submit(torch.tensor([1]), storage.owner, lambda values: None)


def test_query_byte_limit_counts_borrowed_backing_storage():
	service, plane, storage, _ = _setup("sage")
	demand = OnDemand(service, max_bytes=32)
	try:
		nodes = torch.zeros(16, dtype=torch.int64)
		with pytest.raises(AdmissionFull):
			demand.submit(nodes[:1], storage.owner, lambda values: None, borrow=True)
		assert demand.submit(nodes[:1], storage.owner, lambda values: values.shape).result(timeout=10).value == (1, 3)
	finally:
		demand.close()
		plane.close()
