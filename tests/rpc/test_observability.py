from types import SimpleNamespace

import torch

from lib.rpc import inference_pb2 as wire
from lib.rpc.service import OwnedServicer, _to_wire
from tests.graph.test_serving import _setup


def test_completion_and_info_expose_optional_admission_observability():
	service, plane, storage, _ = _setup("sage")
	servicer = OwnedServicer(service, storage.owner)
	context = SimpleNamespace(time_remaining=lambda: 10)
	request = wire.Update(
		kind=wire.Update.INSERT,
		first=_to_wire(torch.tensor([2])),
		second=_to_wire(torch.tensor([0])),
		request_id="rpc-observability",
	)
	try:
		ticket = servicer.Async(request, context)
		completion = servicer.Wait(wire.WaitRequest(ticket=ticket), context)
		assert completion.ready
		assert completion.HasField("admitted_at_ns")
		assert completion.HasField("execution_started_at_ns")
		assert completion.HasField("publication_observed_at_ns")
		assert completion.HasField("completion_at_ns")
		assert completion.HasField("server_reply_prepared_at_ns")
		assert completion.HasField("queue_wait_ns")
		assert completion.HasField("execution_ns")
		assert completion.timestamp_scope.startswith("server perf_counter_ns")
		assert completion.publication_source == "service_completion"
		assert completion.state == "published"
		assert completion.version >= completion.input_version
		info = servicer.Info(wire.Empty(), context)
		assert info.admitted_updates == 1
		assert info.pending_updates == 0
		assert info.pending_update_bytes == 0
	finally:
		servicer.close()
		plane.close()
