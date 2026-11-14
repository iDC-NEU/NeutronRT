from dataclasses import replace

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.delta_plan import MessagePlan, SparseLayerPlan
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.runtime.aggregates.delta import DeltaHeader, pack_delta_compute
from lib.runtime.aggregates.tables import InputTable, unpack_tables


def _packet_inputs():
	plan = SparseLayerPlan(
		torch.tensor([1, 3]),
		MessagePlan(torch.tensor([0, 2]), torch.tensor([1, 0, 1]), torch.tensor([0, 1, 1]), None),
		MessagePlan(torch.tensor([4]), torch.tensor([0]), torch.tensor([0]), torch.tensor([-2])),
	)
	h = ComputeAttempt(StateKey(1, 1, 10), Owner(2, 1), 7)
	a = ComputeAttempt(StateKey(1, 1, 20, "a"), h.owner, 8)
	tables = tuple(
		InputTable(rows, width, values=torch.ones(rows, width))
		for rows, width in ((2, 3), (2, 4), (2, 4), (2, 4), (1, 4))
	)
	return plan, h, a, tables, torch.tensor([3, 2])


def test_delta_packet_preserves_compact_sources_old_values_and_mixed_widths():
	plan, h, a, tables, degrees = _packet_inputs()
	packet = pack_delta_compute(plan, h, a, (4, 3), tables, degrees)
	assert DeltaHeader.decode(packet.header.tensor()) == packet.header
	assert packet.header.input_edges == 3 and packet.header.structural_edges == 1
	assert packet.fields[-6] is degrees and packet.fields[4] is tables[0].values
	layout = PackedLayout(tuple(TensorSpec.of(value) for value in packet.fields))
	assert layout.nbytes == packet.header.nbytes
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for value, view in zip(packet.fields, layout.views(buffer), strict=True):
		view.copy_(value)
	fields, tail = unpack_tables(buffer, 5, extra_specs=packet.header.tail_specs())
	assert fields[0].tolist() == [[2, 3, 0, 1], [2, 4, 0, 1], [2, 4, 0, 1], [2, 4, 0, 1], [1, 4, 0, 1]]
	assert tail[0].tolist() == [1, 10, 8, 20] and tail[-1].tolist() == [-2]


@pytest.mark.parametrize("change", ["old-source", "degree", "route", "multiplicity", "attempt"])
def test_delta_packet_rejects_inconsistent_old_state_routes_and_attempts(change):
	plan, h, a, tables, degrees = _packet_inputs()
	if change == "old-source":
		tables = (*tables[:3], InputTable(1, 4, values=torch.ones(1, 4)), tables[4])
	elif change == "degree":
		degrees[0] = -1
	elif change == "route":
		plan = replace(plan, inputs=replace(plan.inputs, sources=torch.tensor([9, 0, 1])))
	elif change == "multiplicity":
		plan = replace(plan, edges=replace(plan.edges, weights=None))
	else:
		a = replace(a, number=h.number)
	with pytest.raises(ValueError):
		pack_delta_compute(plan, h, a, (4, 3), tables, degrees)
