from dataclasses import replace

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import Location, Owner
from lib.runtime.aggregates.tables import InputTable, pack_tables, table_specs, unpack_tables, validate_fields
from lib.runtime.protocol import InputChunk


def _tables():
	values = torch.arange(16, dtype=torch.float32).reshape(2, 8)[:, ::2]
	chunks = (
		InputChunk(Location(Owner(1, 2), 8), torch.tensor([3, 0]), torch.tensor([2, 0])),
		InputChunk(Location(Owner(2, 1), 9), torch.tensor([4]), torch.tensor([1])),
	)
	return InputTable(2, 4, values=values), InputTable(3, 2, chunks=chunks)


def test_table_packet_keeps_each_source_table_once_and_decodes_borrowed_views():
	tables = _tables()
	packet = pack_tables(tables)
	assert packet.fields[4] is tables[0].values
	assert packet.fields[0].tolist() == [[2, 4, 0, 1], [3, 2, 2, 0]]
	assert packet.fields[1].tolist() == [[1, 1, 2, 8, 2], [1, 2, 1, 9, 1]]
	assert packet.fields[2].tolist() == [3, 0, 4] and packet.fields[3].tolist() == [2, 0, 1]
	tail = torch.tensor([37, 51], dtype=torch.int64)
	layout = PackedLayout((*table_specs(packet.fields[0]), TensorSpec.of(tail)))
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for value, view in zip((*packet.fields, tail), layout.views(buffer), strict=True):
		view.copy_(value)
	fields, extra = unpack_tables(buffer, 2, extra_specs=(TensorSpec.of(tail),))
	for expected, actual in zip(packet.fields, fields, strict=True):
		assert torch.equal(expected, actual)
		if actual.numel():
			assert actual.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
	assert torch.equal(extra[0], tail) and packet.nbytes < layout.nbytes


def test_empty_tables_and_root_proxies_keep_layout_and_generation():
	root = InputTable(0, 4, values=torch.empty(0, 4))
	empty = InputTable(0, 3)
	chunk = InputChunk(Location(Owner(2, 4), 99), torch.tensor([7]), torch.tensor([0]), sender=0)
	packet = pack_tables((root, empty, InputTable(1, 2, chunks=(chunk,))))
	assert packet.fields[1].tolist() == [[2, 0, 4, 99, 1]]
	assert len(validate_fields(packet.fields)) == 3


@pytest.mark.parametrize("change", ["duplicate", "missing", "owner", "dtype", "root_and_chunks"])
def test_invalid_source_tables_are_rejected_before_sending(change):
	root, remote = _tables()
	first, second = remote.chunks
	if change == "duplicate":
		remote = replace(remote, chunks=(replace(first, positions=torch.tensor([0, 0])), second))
	elif change == "missing":
		remote = replace(remote, chunks=(first,))
	elif change == "owner":
		remote = replace(remote, chunks=(replace(first, location=Location(Owner(1, 0), 8)), second))
	elif change == "dtype":
		root = replace(root, values=root.values.double())
	else:
		root = replace(root, chunks=remote.chunks)
	with pytest.raises(ValueError):
		pack_tables((root, remote))


def test_corrupt_descriptor_or_group_identity_cannot_silently_change_the_frame():
	packet = pack_tables(_tables())
	layout = PackedLayout(table_specs(packet.fields[0]))
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for value, view in zip(packet.fields, layout.views(buffer), strict=True):
		view.copy_(value)
	with pytest.raises(ValueError):
		unpack_tables(buffer[:-1], 2)
	with pytest.raises(ValueError):
		unpack_tables(buffer, 3)
	bad = buffer.clone()
	bad[:64].view(torch.int64)[1] = -1
	with pytest.raises(ValueError):
		unpack_tables(bad, 2)
	bad = list(packet.fields)
	bad[1] = bad[1].clone()
	bad[1][0, 0] = 0  # A group's table id no longer matches its descriptor.
	with pytest.raises(ValueError, match="routing"):
		validate_fields(tuple(bad))
