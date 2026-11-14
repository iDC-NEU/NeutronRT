"""Bounded binary CPU tensor payloads for the RPC boundary."""

import sys
import warnings
from dataclasses import dataclass
from math import prod
from threading import Lock
from typing import Tuple

import torch
from torch import Tensor

INT64, FLOAT32, BOOL = 1, 2, 3
_DTYPES = {INT64: torch.int64, FLOAT32: torch.float32, BOOL: torch.bool}
_SIZES = {INT64: 8, FLOAT32: 4, BOOL: 1}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}
_BUFFER_WARNING = Lock()


@dataclass(frozen=True)
class TensorPayload:
	dtype: int
	shape: Tuple[int, ...]
	data: bytes


def encode_tensor(value: Tensor) -> TensorPayload:
	"""Serialize CPU rows once; NumPy packs strided input without a Torch clone."""
	if sys.byteorder != "little":
		raise ValueError("RPC tensors require a little-endian host")
	if value.device.type != "cpu" or value.dtype not in _CODES or value.layout != torch.strided:
		raise ValueError("RPC tensors must use strided CPU int64, float32, or bool storage")
	return TensorPayload(_CODES[value.dtype], tuple(value.shape), value.detach().numpy().tobytes(order="C"))


def decode_tensor(payload: TensorPayload, *, max_bytes: int = 64 * 1024**2) -> Tensor:
	"""Borrow immutable protobuf bytes; consumers must never mutate this view.

	Torch retains the byte object's lifetime. This avoids another input clone at
	the framework boundary; protobuf/network serialization still has its own copies.
	"""
	if sys.byteorder != "little":
		raise ValueError("RPC tensors require a little-endian host")
	if payload.dtype not in _DTYPES or not isinstance(payload.data, bytes) or max_bytes < 0:
		raise ValueError("Invalid RPC tensor dtype, buffer, or byte limit")
	if len(payload.shape) > 8 or any(
		not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= 2**63 - 1 for size in payload.shape
	):
		raise ValueError("Invalid RPC tensor shape")
	count = prod(payload.shape)
	dtype = _DTYPES[payload.dtype]
	if prod(max(1, size) for size in payload.shape) * _SIZES[payload.dtype] > 2**63 - 1:
		raise ValueError("RPC tensor shape would overflow its storage strides")
	if count > 2**63 - 1 or len(payload.data) > max_bytes:
		raise ValueError("RPC tensor exceeds its element or byte limit")
	if count * _SIZES[payload.dtype] != len(payload.data):
		raise ValueError("RPC tensor byte length differs from its shape and dtype")
	if not count:
		return torch.empty(payload.shape, dtype=dtype)
	# These private views are read-only by contract. Torch warns because its
	# general Tensor API also permits writes, which this adapter never performs.
	with _BUFFER_WARNING, warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)
		return torch.frombuffer(payload.data, dtype=dtype).reshape(payload.shape)
