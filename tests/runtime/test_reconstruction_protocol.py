from dataclasses import replace

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt, Location, Owner, StateKey
from lib.runtime.aggregates.reconstruction import ReconstructionHeader, pack_reconstruction
from lib.runtime.aggregates.tables import InputTable, unpack_tables
from lib.runtime.protocol import InputChunk


def _recipe(start=0, rows=3):
	attempt = ComputeAttempt(StateKey(7, 2, 33), Owner(2, 1), 9)
	widths = (4, 5, 3)
	tables = tuple(
		InputTable(rows, width, values=torch.ones(rows, width))
		if layer == 0
		else InputTable(
			rows,
			width,
			chunks=(InputChunk(Location(Owner(1, 1), layer), torch.arange(rows), torch.arange(rows)),) if rows else (),
		)
		for layer, width in enumerate(widths)
		if layer >= start
	)
	return attempt, widths, tables, torch.arange(rows, dtype=torch.int64)


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("rows", [0, 3])
def test_reconstruction_wire_preserves_borrowed_inputs_and_cached_suffix(start, rows):
	attempt, widths, tables, degrees = _recipe(start, rows)
	packet = pack_reconstruction(attempt, widths, tables, degrees, start_layer=start)
	assert ReconstructionHeader.decode(packet.header.tensor()) == packet.header
	assert packet.fields[-1] is degrees
	if start == 0:
		assert packet.fields[4] is tables[0].values
	layout = PackedLayout(tuple(TensorSpec.of(field) for field in packet.fields))
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for field, view in zip(packet.fields, layout.views(buffer), strict=True):
		view.copy_(field)
	fields, extra = unpack_tables(buffer, packet.header.count, extra_specs=packet.header.tail_specs())
	assert extra[0].tolist() == [7, 33] and torch.equal(extra[1], degrees)
	if start:
		assert fields[0].tolist() == [[rows, width, int(rows > 0), 0] for width in widths[start:]]
	else:
		assert fields[0][0].tolist() == [rows, 4, 0, 1]


@pytest.mark.parametrize("change", ["range", "width", "root", "degree", "kind"])
def test_reconstruction_rejects_inconsistent_recipes(change):
	attempt, widths, tables, degrees = _recipe()
	if change == "range":
		widths = widths[:2]
	elif change == "width":
		tables = (*tables[:-1], replace(tables[-1], width=8))
	elif change == "root":
		tables = (replace(tables[0], values=None, chunks=tables[1].chunks), *tables[1:])
	elif change == "degree":
		degrees[0] = -1
	else:
		attempt = replace(attempt, key=replace(attempt.key, kind="a"))
	with pytest.raises(ValueError):
		pack_reconstruction(attempt, widths, tables, degrees)
