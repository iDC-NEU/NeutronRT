import gc
from dataclasses import replace

import pytest
import torch

from lib.rpc.tensors import BOOL, FLOAT32, INT64, TensorPayload, decode_tensor, encode_tensor


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bool])
def test_binary_payload_preserves_strided_rows_scalars_and_empty_shapes(dtype):
	base = torch.arange(24).reshape(4, 6).to(dtype)
	for value in (base[:, ::2], base[0, 0], base[:0], torch.empty(2, 0, 3, dtype=dtype)):
		payload = encode_tensor(value)
		actual = decode_tensor(payload)
		assert actual.shape == value.shape and actual.dtype == value.dtype
		assert torch.equal(actual, value)
		if value.numel():
			assert actual.data_ptr() == decode_tensor(payload).data_ptr()
			del payload
			gc.collect()
			assert torch.equal(actual, value)


def test_rpc_payload_preserves_large_integer_and_float_bit_patterns():
	for value in (
		torch.tensor([2**60 + 3, -(2**60 + 7)], dtype=torch.int64),
		torch.tensor([float("nan"), -0.0, float("inf"), 1.5]),
	):
		actual = decode_tensor(encode_tensor(value))
		assert torch.equal(actual.view(torch.uint8), value.view(torch.uint8))


@pytest.mark.parametrize(
	"payload",
	[
		TensorPayload(0, (1,), b"1234"),
		TensorPayload(FLOAT32, (2,), b"1234"),
		TensorPayload(INT64, (-1,), b""),
		TensorPayload(BOOL, (True,), b"0"),
		TensorPayload(BOOL, (1,) * 9, b"0"),
		TensorPayload(BOOL, (2**63,), b""),
		TensorPayload(FLOAT32, (2**32, 2**32), b""),
		TensorPayload(FLOAT32, (0, 2**62), b""),
	],
)
def test_bad_wire_shapes_and_lengths_fail_before_allocating_tensor_storage(payload):
	with pytest.raises(ValueError):
		decode_tensor(payload)


def test_wire_byte_budget_is_checked_before_borrowing_the_input():
	payload = encode_tensor(torch.zeros(8))
	with pytest.raises(ValueError, match="limit"):
		decode_tensor(payload, max_bytes=31)
	with pytest.raises(ValueError):
		decode_tensor(replace(payload, dtype=0))
