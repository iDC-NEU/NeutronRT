from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lib.profiling.experiment import ProfileCase, make_trace
from lib.rpc import inference_pb2 as wire
from scripts.evaluate.profile import exercise


class _Stub:
	def __init__(self):
		self.requests = {}
		self.active = self.peak = self.queries = self.completed = self.releases = 0

	def Async(self, request, **_kwargs):
		ticket = wire.Ticket(session="test", number=len(self.requests) + 1, request_id=request.request_id)
		self.requests[ticket.number] = request
		self.active += 1
		self.peak = max(self.active, self.peak)
		return ticket

	def Sync(self, request, **kwargs):
		return self.Wait(wire.WaitRequest(ticket=self.Async(request)), **kwargs)

	def Wait(self, request, **_kwargs):
		self.completed += 1
		return wire.Completion(
			ticket=request.ticket,
			ready=True,
			version=self.completed,
			input_changes=self.requests[request.ticket.number].first.shape[0],
		)

	def Release(self, _ticket, **_kwargs):
		self.active -= 1
		self.releases += 1

	def Query(self, _request, **_kwargs):
		self.queries += 1
		return SimpleNamespace(version=self.completed)


@pytest.mark.parametrize("interface", ["sync", "async"])
def test_client_enforces_inflight_limits_waits_for_all_publications_and_releases_tickets(interface):
	stub = _Stub()
	trace = make_trace(torch.zeros(20, 4), updates=8, changes=2, pool_size=8)
	case = ProfileCase("case", interface=interface, max_inflight=3, query_every=2)
	report = exercise(stub, trace, case, "test", torch.tensor([1, 2]), prefix="unique", timeout=10)
	assert stub.peak == (1 if interface == "sync" else 3)
	assert stub.completed == stub.releases == report["completed_updates"] == 8
	assert stub.active == 0 and stub.queries == report["query_latency"]["count"] == 4
	assert report["completed_input_changes"] == 16
	assert len(report["raw_completion_seconds"]) == 8
	assert report["admission_latency"]["count"] == (0 if interface == "sync" else 8)
	assert {request.request_id for request in stub.requests.values()} == {f"unique-{index}" for index in range(8)}
	assert (
		exercise(stub, [], replace(case, query_every=0), "test", torch.tensor([1]), prefix="empty", timeout=10)[
			"client_changes_per_second"
		]
		is None
	)
