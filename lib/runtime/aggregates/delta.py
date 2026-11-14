"""Compact projected-sum updates with distinct old/new source tables."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.delta_plan import MessagePlan, SparseLayerPlan
from lib.graph.ownership import ComputeAttempt
from lib.models.aggregates import delta_sage_workspace_bytes
from lib.runtime.aggregates.tables import InputTable, layouts, pack_tables
from lib.runtime.protocol import DELTA_COMPUTE


@dataclass(frozen=True)
class DeltaHeader:
	request: int
	generation: int
	layer: int
	targets: int
	input_edges: int
	structural_edges: int
	nbytes: int

	def tensor(self) -> Tensor:
		return torch.tensor(
			[
				DELTA_COMPUTE,
				self.request,
				self.generation,
				self.layer,
				self.targets,
				self.input_edges,
				self.structural_edges,
				self.nbytes,
			],
			dtype=torch.int64,
		)

	@classmethod
	def decode(cls, value: Tensor) -> "DeltaHeader":
		if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != (8,):
			raise ValueError("Sparse delta commands require eight CPU int64 fields")
		opcode, *fields = value.tolist()
		header = cls(*fields)
		if (
			opcode != DELTA_COMPUTE
			or min(header.request, header.generation, header.layer) < 1
			or min(header.targets, header.input_edges, header.structural_edges) < 0
			or header.nbytes < 192 + (header.targets + 2 * header.input_edges + 3 * header.structural_edges) * 8
		):
			raise ValueError("Invalid sparse delta identity, dimensions or frame size")
		return header

	def tail_specs(self) -> Tuple[TensorSpec, ...]:
		return (
			TensorSpec((4,), torch.int64),
			TensorSpec((self.targets,), torch.int64),
			TensorSpec((self.input_edges,), torch.int64),
			TensorSpec((self.input_edges,), torch.int64),
			TensorSpec((self.structural_edges,), torch.int64),
			TensorSpec((self.structural_edges,), torch.int64),
			TensorSpec((self.structural_edges,), torch.int64),
		)


@dataclass(frozen=True)
class DeltaPacket:
	header: DeltaHeader
	fields: Tuple[Tensor, ...]


@dataclass(frozen=True)
class DeltaWorkspace:
	frame_bytes: int
	required_bytes: int


def delta_workspace(
	plan: SparseLayerPlan, widths: Sequence[int], layer: int, tables: Sequence[InputTable]
) -> DeltaWorkspace:
	"""Reserve receive frames, resolved inputs and model scratch before input I/O."""
	sizes = [table.rows * table.width * 4 for table in tables]
	routed_rows = sum(table.rows for table in tables if table.values is None)
	groups = sum(len(table.chunks) for table in tables)
	root_values = sum(size for table, size in zip(tables, sizes, strict=True) if table.values is not None)
	input_edges, structural_edges = plan.inputs.sources.numel(), plan.edges.sources.numel()
	frame = (
		192
		+ 128
		+ 40 * groups
		+ 16 * routed_rows
		+ root_values
		+ 8 * plan.targets.numel()
		+ 16 * input_edges
		+ 24 * structural_edges
	)
	model = delta_sage_workspace_bytes(
		plan.targets.numel(),
		plan.inputs.value_nodes.numel(),
		plan.edges.value_nodes.numel(),
		widths[layer - 1],
		widths[layer],
		input_edges + structural_edges,
	)
	# Input resolution may hold one grouped receive alongside all resolved tables.
	return DeltaWorkspace(frame, frame + sum(sizes) + max(sizes, default=0) + model)


def validate_delta_tables(header: DeltaHeader, fields: Tuple[Tensor, ...], widths: Sequence[int]) -> None:
	shapes = layouts(fields[0])
	if len(shapes) != 5 or header.layer >= len(widths):
		raise ValueError("Sparse deltas need five tables at a valid model layer")
	if any(
		shape.width != widths[header.layer if index == 0 else header.layer - 1] for index, shape in enumerate(shapes)
	):
		raise ValueError("Sparse tables do not match projected-sum and layer-input widths")
	if shapes[0].rows != header.targets or shapes[1].rows != header.targets or shapes[2].rows != shapes[3].rows:
		raise ValueError("Sparse tables do not align target rows or old/new changed sources")


def _route(message: MessagePlan, count: int, *, structural: bool) -> None:
	for value in (message.value_nodes, message.sources, message.targets):
		if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1:
			raise ValueError("Sparse routes require CPU int64 node/index vectors")
	if (
		message.targets.shape != message.sources.shape
		or message.sources.numel()
		and (
			message.sources.min() < 0
			or message.sources.max() >= message.value_nodes.numel()
			or message.targets.min() < 0
			or message.targets.max() >= count
		)
	):
		raise ValueError("Sparse route indices are outside compact source/target tables")
	if structural:
		if (
			message.weights is None
			or message.weights.dtype != torch.int64
			or message.weights.device.type != "cpu"
			or message.weights.shape != message.sources.shape
		):
			raise ValueError("Structural routes require aligned signed int64 multiplicities")
	elif message.weights is not None:
		raise ValueError("Input delta routes currently encode parallel edges as repeated indices")


def pack_delta_compute(
	plan: SparseLayerPlan,
	embedding: ComputeAttempt,
	aggregate: ComputeAttempt,
	widths: Sequence[int],
	tables: Sequence[InputTable],
	old_degrees: Tensor,
) -> DeltaPacket:
	if (
		embedding.key.kind != "h"
		or aggregate.key.kind != "a"
		or embedding.owner != aggregate.owner
		or embedding.key.version != aggregate.key.version
		or embedding.key.layer != aggregate.key.layer
		or min(embedding.number, aggregate.number) < 1
		or embedding.number == aggregate.number
	):
		raise ValueError("Sparse outputs need distinct same-owner/version/layer h and aggregate attempts")
	count = plan.targets.numel()
	if (
		plan.targets.dtype != torch.int64
		or plan.targets.device.type != "cpu"
		or plan.targets.ndim != 1
		or plan.targets.unique().numel() != count
	):
		raise ValueError("Sparse output nodes require a unique CPU int64 vector")
	if (
		old_degrees.dtype != torch.int64
		or old_degrees.device.type != "cpu"
		or old_degrees.shape != (count,)
		or (old_degrees < 0).any()
	):
		raise ValueError("Sparse old degrees must align nonnegative target rows")
	_route(plan.inputs, count, structural=False)
	_route(plan.edges, count, structural=True)
	assert plan.edges.weights is not None
	inputs = pack_tables(tables)
	fields = (
		*inputs.fields,
		torch.tensor(
			[embedding.key.version, embedding.key.chunk, aggregate.number, aggregate.key.chunk], dtype=torch.int64
		),
		old_degrees,
		plan.inputs.sources,
		plan.inputs.targets,
		plan.edges.sources,
		plan.edges.targets,
		plan.edges.weights,
	)
	header = DeltaHeader(
		embedding.number,
		embedding.owner.generation,
		embedding.key.layer,
		count,
		plan.inputs.sources.numel(),
		plan.edges.sources.numel(),
		PackedLayout(tuple(TensorSpec.of(value) for value in fields)).nbytes,
	)
	DeltaHeader.decode(header.tensor())
	validate_delta_tables(header, inputs.fields, widths)
	shapes = layouts(inputs.fields[0])
	if shapes[2].rows != plan.inputs.value_nodes.numel() or shapes[4].rows != plan.edges.value_nodes.numel():
		raise ValueError("Sparse source tables do not follow their compact route nodes")
	return DeltaPacket(header, fields)
