import pytest
import torch

from lib.rpc import inference_pb2 as wire
from lib.rpc.service import OwnedServicer, _to_wire
from tests.graph.test_serving import _setup


@pytest.mark.parametrize("deferred", [False, True])
def test_rpc_attributes_survive_admission_and_participate_in_request_deduplication(deferred):
	service, plane, storage, _ = _setup("sage")
	service.graph.edata["nrt"] = torch.zeros(service.graph.num_edges(), 4)
	if deferred:
		from lib.graph.deferred import DeferredUpdates

		DeferredUpdates(service, max_versions=2, idle_seconds=60)
	frontend = OwnedServicer(service, storage.owner)
	try:
		attributes = torch.tensor([[2.0, 0.8, 0.1, -0.2]])
		request = wire.Update(
			kind=wire.Update.INSERT,
			first=_to_wire(torch.tensor([2])),
			second=_to_wire(torch.tensor([0])),
			attributes=_to_wire(attributes),
			request_id="typed",
		)
		entry = frontend._admit(request, strict=True)
		assert entry.result(timeout=10).version == 1
		torch.testing.assert_close(service.graph.edata["nrt"][-1:], attributes)
		assert frontend._admit(request, strict=True) is entry
		attributes[:, 1] += 0.1
		request.attributes.CopyFrom(_to_wire(attributes))
		with pytest.raises(ValueError, match="different"):
			frontend._admit(request)
		assert service.graph.num_edges() == service.graph.edata["nrt"].shape[0]
	finally:
		frontend.close()
		plane.close()
