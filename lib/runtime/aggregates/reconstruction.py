"""Row-only GraphSAGE reconstruction recipes; topology stays at storage."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt
from lib.runtime.aggregates.tables import InputTable, layouts, pack_tables
from lib.runtime.protocol import RECONSTRUCT


@dataclass(frozen=True)
class ReconstructionHeader:
	request: int
	generation: int
	layer: int
	start_layer: int
	rows: int
	nbytes: int

	@property
	def targets(self) -> int:
		return self.rows

	@property
	def count(self) -> int:
		return self.layer - self.start_layer + 1

	def tensor(self) -> Tensor:
		return torch.tensor(
			[RECONSTRUCT, self.request, self.generation, self.layer, self.start_layer, self.rows, self.nbytes, 0],
			dtype=torch.int64,
		)

	@classmethod
	def decode(cls, value: Tensor) -> "ReconstructionHeader":
		if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != (8,):
			raise ValueError("Reconstruction headers require eight CPU int64 fields")
		opcode, request, generation, layer, start, rows, nbytes, reserved = value.tolist()
		if (
			opcode != RECONSTRUCT
			or reserved != 0
			or min(request, generation, layer) < 1
			or not 0 <= start < layer
			or rows < 0
			or nbytes < (layer - start + 1) * 32 + 16 + rows * 8
		):
			raise ValueError("Invalid reconstruction identity, layer range or frame size")
		return cls(request, generation, layer, start, rows, nbytes)

	def tail_specs(self) -> Tuple[TensorSpec, ...]:
		return (TensorSpec((2,), torch.int64), TensorSpec((self.rows,), torch.int64))


@dataclass(frozen=True)
class ReconstructionPacket:
	header: ReconstructionHeader
	fields: Tuple[Tensor, ...]


def validate_recipe(header: ReconstructionHeader, fields: Tuple[Tensor, ...], widths: Sequence[int]) -> None:
	"""Shapes certify layout only; the coordinator must certify dependency freshness."""
	shapes = layouts(fields[0])
	if header.layer >= len(widths) or len(shapes) != header.count:
		raise ValueError("Reconstruction tables do not match the requested model range")
	if any(
		shape.rows != header.rows or shape.width != width
		for shape, width in zip(shapes, widths[header.start_layer : header.layer + 1], strict=True)
	):
		raise ValueError("Reconstruction tables must align selected nodes and model widths")
	if shapes[0].root != (header.start_layer == 0) or any(shape.root for shape in shapes[1:]):
		raise ValueError("Reconstruction starts with features or cached h, followed by routed projected sums")


def pack_reconstruction(
	attempt: ComputeAttempt,
	widths: Sequence[int],
	tables: Sequence[InputTable],
	degrees: Tensor,
	*,
	start_layer: int = 0,
) -> ReconstructionPacket:
	if attempt.key.kind != "h" or min(attempt.number, attempt.owner.generation) < 1:
		raise ValueError("Reconstruction requires an embedding attempt")
	if type(start_layer) is not int or not 0 <= start_layer < attempt.key.layer < len(widths):
		raise ValueError("Reconstruction layer range is outside the model")
	if degrees.device.type != "cpu" or degrees.dtype != torch.int64 or degrees.ndim != 1 or (degrees < 0).any():
		raise ValueError("Reconstruction degrees require a nonnegative CPU int64 vector")
	inputs = pack_tables(tables)
	fields = (
		*inputs.fields,
		torch.tensor([attempt.key.version, attempt.key.chunk], dtype=torch.int64),
		degrees,
	)
	header = ReconstructionHeader(
		attempt.number,
		attempt.owner.generation,
		attempt.key.layer,
		start_layer,
		degrees.numel(),
		PackedLayout(tuple(TensorSpec.of(field) for field in fields)).nbytes,
	)
	ReconstructionHeader.decode(header.tensor())
	validate_recipe(header, inputs.fields, widths)
	return ReconstructionPacket(header, fields)
