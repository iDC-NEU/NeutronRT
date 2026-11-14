import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec


def test_mixed_payload_alignment_and_integer_bits():
	values = [
		torch.tensor([True, False]),
		torch.tensor([2**60 + 7, -(2**60) + 9]),
		torch.tensor([float("inf"), -0.0, 1.5]),
		torch.tensor(3.5),
		torch.empty(0, 4),
	]
	layout = PackedLayout(TensorSpec.of(value) for value in values)
	buffer = torch.zeros(layout.nbytes, dtype=torch.uint8)
	for value, (start, end) in zip(values, layout.slices, strict=True):
		assert start % value.element_size() == 0
		buffer[start:end].copy_(value.reshape(-1).view(torch.uint8))
	for expected, actual in zip(values, layout.views(buffer), strict=True):
		assert torch.equal(expected.reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8))


def test_packed_layout_rejects_wrong_buffer_and_negative_shapes():
	with pytest.raises(ValueError, match="Negative"):
		PackedLayout([TensorSpec((-1, 3), torch.float32)])
	layout = PackedLayout([TensorSpec((3,), torch.int64)])
	with pytest.raises(ValueError, match="match"):
		layout.views(torch.empty(3, dtype=torch.uint8))


def test_version_dimensions_and_buffer_alignment_are_checked():
	with pytest.raises(ValueError, match="version"):
		PackedLayout([], version=2)
	with pytest.raises(ValueError, match="integers"):
		PackedLayout([TensorSpec((1.5,), torch.float32)])
	layout = PackedLayout([TensorSpec((1,), torch.int64)])
	with pytest.raises(ValueError, match="aligned"):
		layout.views(torch.empty(9, dtype=torch.uint8)[1:])
	with pytest.raises(ValueError, match="match"):
		layout.views(torch.empty(16, dtype=torch.uint8)[::2])


def test_received_fields_share_the_single_transport_buffer():
	layout = PackedLayout([TensorSpec((3,), torch.bool), TensorSpec((2,), torch.int64), TensorSpec((), torch.float32)])
	buffer = torch.zeros(layout.nbytes, dtype=torch.uint8)
	values = layout.views(buffer)
	values[0].fill_(True)
	values[1].copy_(torch.tensor([2**60 + 1, -7]))
	values[2].fill_(1.5)
	assert all(value.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr() for value in values)
	assert layout.views(buffer)[1].tolist() == [2**60 + 1, -7]
	assert layout.version == 1
