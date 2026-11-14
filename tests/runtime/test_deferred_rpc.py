"""Exercise frontend contracts in memory; real socket cases run in the host batch."""

import grpc
import pytest
import torch

from lib.graph.deferred import DeferredUpdates
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import OwnedServicer, _to_wire
from lib.rpc.tensors import TensorPayload, decode_tensor
from tests.graph.test_serving import _current_reference, _setup


class _Abort(Exception):
	def __init__(self, code, message):
		self.code = code
		super().__init__(message)


class _Context:
	def time_remaining(self):
		return 10

	def abort(self, code, message):
		raise _Abort(code, message)


def _values(response):
	return decode_tensor(TensorPayload(response.values.dtype, tuple(response.values.shape), response.values.data))


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_stale_reads_report_one_budget_and_sync_retry_flushes_without_duplicate_mutation(name):
	service, plane, storage, initial = _setup(name, batch_size=2)
	service.ensure(torch.arange(6))
	deferred = DeferredUpdates(service, max_versions=2, idle_seconds=60)
	frontend = OwnedServicer(service, storage.owner)
	context = _Context()
	try:
		requests = [
			wire.Update(
				kind=wire.Update.FEATURES,
				first=_to_wire(torch.tensor([1])),
				second=_to_wire(service.features[[1]] + offset),
				request_id=f"input-{offset}",
			)
			for offset in (1, 2)
		]
		tickets = [frontend.Async(request, context) for request in requests]
		frontend.admission._pool.submit(lambda: None).result(5)
		info = frontend.Info(wire.Empty(), context)
		assert info.graph_version == 2 and info.completed_updates == 0 and info.max_staleness == 2
		assert not frontend.Wait(wire.WaitRequest(ticket=tickets[0], poll=True), context).ready
		nodes = torch.tensor([1, 2, 3, 1])
		response = frontend.Query(wire.QueryRequest(nodes=_to_wire(nodes), max_staleness=2), context)
		assert response.version == 2 and response.max_age == 2 and response.min_fresh_version == 0
		torch.testing.assert_close(_values(response), initial[nodes])
		features = frontend.Query(wire.QueryRequest(nodes=_to_wire(nodes), layer=0, max_staleness=2), context)
		assert features.max_age == 0 and features.min_fresh_version == 2
		torch.testing.assert_close(_values(features), service.features[nodes])
		with pytest.raises(_Abort) as aborted:
			frontend.Query(wire.QueryRequest(nodes=_to_wire(nodes), max_staleness=3), context)
		assert aborted.value.code == grpc.StatusCode.INVALID_ARGUMENT
		fresh = frontend.Query(wire.QueryRequest(nodes=_to_wire(nodes)), context)
		assert fresh.max_age == 0 and fresh.min_fresh_version == 2
		torch.testing.assert_close(_values(fresh), _current_reference(service)[nodes], rtol=2e-5, atol=2e-5)
		completion = frontend.Sync(requests[0], context)
		other = frontend.Wait(wire.WaitRequest(ticket=tickets[1]), context)
		assert completion.version == other.version == 2
		assert (completion.input_version, other.input_version) == (1, 2)
		assert service.catalog.version == service.counters.updates == service.counters.input_changes == 2
	finally:
		frontend.close()
		deferred.close()
		plane.close()


def test_positive_query_budget_is_rejected_when_deferred_serving_is_disabled():
	service, plane, storage, _ = _setup("sage")
	frontend = OwnedServicer(service, storage.owner)
	try:
		with pytest.raises(_Abort) as aborted:
			frontend.Query(wire.QueryRequest(nodes=_to_wire(torch.tensor([1])), max_staleness=1), _Context())
		assert aborted.value.code == grpc.StatusCode.INVALID_ARGUMENT
	finally:
		frontend.close()
		plane.close()
