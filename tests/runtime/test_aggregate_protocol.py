from dataclasses import replace

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.runtime.aggregates.protocol import aggregate_identity, aggregate_specs, pack_aggregate_compute
from lib.runtime.protocol import COMPUTE_WITH_AGGREGATE, ComputeHeader
from tests.runtime.test_protocol import _attempt, _task


def test_pair_packet_keeps_one_input_body_and_distinct_output_identities():
	h = _attempt(1)
	a = ComputeAttempt(StateKey(7, 1, 99, "a"), h.owner, 10)
	features = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
	packet = pack_aggregate_compute(_task(1), h, a, width=4, features=features)
	assert packet.fields[9] is features
	assert (
		ComputeHeader.decode(packet.header.tensor(opcode=COMPUTE_WITH_AGGREGATE), opcode=COMPUTE_WITH_AGGREGATE)
		== packet.header
	)
	assert tuple(TensorSpec.of(field) for field in packet.fields) == aggregate_specs(packet.header, 4)
	first, second = packet.result_packets()
	assert first is packet.embedding and first.header.request == 9 and second.header.request == 10
	assert first.fields[0].tolist() == [7, 33, 0] and second.fields[0].tolist() == [7, 99, 0]
	assert all(before is after for before, after in zip(first.fields[1:], second.fields[1:], strict=True))
	layout = PackedLayout(aggregate_specs(packet.header, 4))
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for field, view in zip(packet.fields, layout.views(buffer), strict=True):
		view.copy_(field)
	for field, view in zip(packet.fields, layout.views(buffer), strict=True):
		assert torch.equal(field, view)


@pytest.mark.parametrize("change", ["kind", "owner", "version", "layer", "request"])
def test_pair_packet_rejects_cross_state_attempts(change):
	h = _attempt(1)
	a = ComputeAttempt(StateKey(7, 1, 99, "a"), h.owner, 10)
	if change == "owner":
		a = replace(a, owner=Owner(1, 1))
	elif change == "request":
		a = replace(a, number=h.number)
	else:
		key = replace(a.key, **{change: {"kind": "h", "version": 8, "layer": 2}[change]})
		a = replace(a, key=key)
	with pytest.raises(ValueError, match="Paired"):
		pack_aggregate_compute(_task(1), h, a, width=4, features=torch.ones(3, 4))


def test_receiver_rejects_inconsistent_aggregate_identity_after_full_body():
	h = _attempt(1)
	a = ComputeAttempt(StateKey(7, 1, 99, "a"), h.owner, 10)
	packet = pack_aggregate_compute(_task(1), h, a, width=4, features=torch.ones(3, 4))
	for identity in (torch.tensor([9, 7, 1, 99]), torch.tensor([10, 8, 1, 99]), torch.tensor([10, 7, 2, 99])):
		with pytest.raises(ValueError, match="Aggregate identity"):
			aggregate_identity(packet.header, (*packet.fields[:-1], identity))
