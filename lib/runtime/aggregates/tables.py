"""Compact row-table inputs for aggregate reconstruction and sparse updates.

Each value table is represented once, with selected chunk offsets and placement
metadata. Source vectors are never repeated per destination edge here. The first
field describes the layout so a packed frame can be decoded without another
metadata round trip. Float values and int64 routing bits retain their dtypes.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import Owner
from lib.runtime.protocol import ComputeHeader, InputChunk

if TYPE_CHECKING:
	from lib.runtime.data_plane import RemoteEmbeddingPlane
	from lib.runtime.worker import OwnedWorker


@dataclass(frozen=True)
class InputTable:
	rows: int
	width: int
	values: Optional[Tensor] = None
	chunks: Tuple[InputChunk, ...] = ()


@dataclass(frozen=True)
class TableLayout:
	rows: int
	width: int
	groups: int
	root: bool


def layouts(descriptor: Tensor) -> Tuple[TableLayout, ...]:
	if descriptor.dtype != torch.int64 or descriptor.ndim != 2 or descriptor.shape[1] != 4 or not descriptor.shape[0]:
		raise ValueError("Table descriptor requires one int64 [rows,width,groups,root] row per table")
	result = []
	for rows, width, groups, root in descriptor.cpu().tolist():
		if min(rows, groups) < 0 or width < 1 or root not in (0, 1):
			raise ValueError("Invalid row-table dimensions or root flag")
		if root and groups or not root and ((rows > 0) != (groups > 0)) or groups > rows:
			raise ValueError("Row-table groups do not match its source layout")
		result.append(TableLayout(rows, width, groups, bool(root)))
	return tuple(result)


def table_specs(descriptor: Tensor) -> Tuple[TensorSpec, ...]:
	shapes = layouts(descriptor)
	groups = sum(shape.groups for shape in shapes)
	rows = sum(shape.rows for shape in shapes if not shape.root)
	return (
		TensorSpec(tuple(descriptor.shape), torch.int64),
		TensorSpec((groups, 5), torch.int64),
		TensorSpec((rows,), torch.int64),
		TensorSpec((rows,), torch.int64),
		*(TensorSpec((shape.rows if shape.root else 0, shape.width), torch.float32) for shape in shapes),
	)


@dataclass(frozen=True)
class TablePacket:
	fields: Tuple[Tensor, ...]

	@property
	def nbytes(self) -> int:
		return int(PackedLayout(table_specs(self.fields[0])).nbytes)


def _indices(value: Tensor) -> None:
	if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1:
		raise ValueError("Table row routing requires one-dimensional CPU int64 indices")


def pack_tables(tables: Sequence[InputTable]) -> TablePacket:
	if not tables:
		raise ValueError("A table packet needs at least one input table")
	descriptor, metadata, row_parts, position_parts, features = [], [], [], [], []
	for index, table in enumerate(tables):
		if type(table.rows) is not int or type(table.width) is not int or table.rows < 0 or table.width < 1:
			raise ValueError("Table rows/width must be nonnegative/positive integers")
		if table.values is not None:
			if (
				table.chunks
				or table.values.device.type != "cpu"
				or table.values.dtype != torch.float32
				or table.values.shape != (table.rows, table.width)
			):
				raise ValueError("Root table values require matching CPU float32 rows and no chunk routes")
			features.append(table.values)
		else:
			features.append(torch.empty(0, table.width, dtype=torch.float32))
			positions = []
			for chunk in table.chunks:
				_indices(chunk.rows)
				_indices(chunk.positions)
				if chunk.positions.shape != chunk.rows.shape or not chunk.rows.numel() or chunk.rows.min() < 0:
					raise ValueError("Table chunks need nonempty aligned row offsets and destination positions")
				owner = chunk.location.owner
				if owner.rank < 0 or owner.generation < 1 or chunk.location.handle < 0:
					raise ValueError("Table chunk has an invalid source owner or handle")
				if chunk.sender is not None and (type(chunk.sender) is not int or chunk.sender != 0):
					raise ValueError("A table proxy sender must identify the storage coordinator")
				metadata.append(
					[
						index,
						owner.rank if chunk.sender is None else chunk.sender,
						owner.generation,
						chunk.location.handle,
						chunk.rows.numel(),
					]
				)
				row_parts.append(chunk.rows)
				position_parts.append(chunk.positions)
				positions.append(chunk.positions)
			covered = torch.cat(positions) if positions else torch.empty(0, dtype=torch.int64)
			if not torch.equal(covered.sort().values, torch.arange(table.rows)):
				raise ValueError("Chunk destinations must partition every row of its table")
		descriptor.append([table.rows, table.width, len(table.chunks), int(table.values is not None)])
	packet = TablePacket(
		(
			torch.tensor(descriptor, dtype=torch.int64),
			torch.tensor(metadata, dtype=torch.int64).reshape(-1, 5),
			torch.cat(row_parts) if row_parts else torch.empty(0, dtype=torch.int64),
			torch.cat(position_parts) if position_parts else torch.empty(0, dtype=torch.int64),
			*features,
		)
	)
	validate_fields(packet.fields)
	return packet


def validate_fields(fields: Tuple[Tensor, ...]) -> Tuple[TableLayout, ...]:
	if not fields:
		raise ValueError("Missing table descriptor")
	shapes = layouts(fields[0])
	if tuple(TensorSpec.of(field) for field in fields) != table_specs(fields[0]):
		raise ValueError("Table fields do not match the declared layout")
	metadata = fields[1].cpu().tolist()
	group_offset = 0
	for index, shape in enumerate(shapes):
		groups = metadata[group_offset : group_offset + shape.groups]
		if any(
			table != index or rank < 0 or generation < 1 or handle < 0 or count < 1
			for table, rank, generation, handle, count in groups
		):
			raise ValueError("Table routing has an invalid table index, source identity or row count")
		if sum(group[4] for group in groups) != (0 if shape.root else shape.rows):
			raise ValueError("Table routing row counts do not match its descriptor")
		group_offset += shape.groups
	return shapes


def unpack_tables(
	buffer: Tensor, count: int, *, extra_specs: Sequence[TensorSpec] = ()
) -> tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
	"""Decode borrowed views, including an operator's separately described tail."""
	if type(count) is not int or count < 1 or buffer.dtype != torch.uint8 or buffer.ndim != 1:
		raise ValueError("A packed table frame needs a positive table count and byte vector")
	prefix_bytes = count * 4 * 8
	if buffer.numel() < prefix_bytes or not buffer.is_contiguous() or buffer.storage_offset() % 8:
		raise ValueError("Packed table descriptor is truncated or misaligned")
	descriptor = buffer[:prefix_bytes].view(torch.int64).reshape(count, 4)
	specs = table_specs(descriptor)
	views = PackedLayout((*specs, *extra_specs)).views(buffer)
	fields, extra = tuple(views[: len(specs)]), tuple(views[len(specs) :])
	validate_fields(fields)
	return fields, extra


def receive_tables(
	worker: "OwnedWorker", nbytes: int, count: int, *, extra_specs: Sequence[TensorSpec] = ()
) -> tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
	if type(count) is not int or count < 1 or type(nbytes) is not int or nbytes < count * 4 * 8:
		raise ValueError("A received table frame needs a positive descriptor count and sufficient bytes")
	if worker.coalesced:
		buffer = worker.comm.recv_tensor(nbytes, dtype=torch.uint8, src=0, name="aggregate-tables")
		return unpack_tables(buffer, count, extra_specs=extra_specs)
	descriptor = worker.comm.recv_tensor((count, 4), dtype=torch.int64, src=0, name="table-descriptor")
	specs = table_specs(descriptor)
	all_specs = (*specs, *extra_specs)
	if PackedLayout(all_specs).nbytes != nbytes:
		raise ValueError("Typed table frame size does not match its declared layout")
	values = [descriptor]
	for spec in all_specs[1:]:
		values.append(
			worker.comm.recv_tensor(spec.shape, dtype=spec.dtype, src=0)
			if all(spec.shape)
			else worker.comm.empty(spec.shape, dtype=spec.dtype)
		)
	fields, extra = tuple(values[: len(specs)]), tuple(values[len(specs) :])
	validate_fields(fields)
	return fields, extra


def resolve_tables(worker: "OwnedWorker", fields: Tuple[Tensor, ...], stack: ExitStack) -> Tuple[Tensor, ...]:
	"""Reuse device-ready input resolution at each table's actual width."""
	shapes = validate_fields(fields)
	blank = worker.comm.empty((0,), dtype=torch.int64)
	group_offset = row_offset = 0
	result = []
	for index, shape in enumerate(shapes):
		routed = 0 if shape.root else shape.rows
		groups = fields[1][group_offset : group_offset + shape.groups, 1:]
		rows = fields[2][row_offset : row_offset + routed]
		positions = fields[3][row_offset : row_offset + routed]
		header = ComputeHeader(1, worker.owner.generation, 1 if shape.root else 2, shape.rows, 0, 0, shape.groups)
		inputs = (*((blank,) * 6), groups, rows, positions, fields[4 + index])
		result.append(worker._inputs(header, inputs, stack, width=shape.width))
		group_offset += shape.groups
		row_offset += routed
	return tuple(result)


def supply_tables(
	plane: "RemoteEmbeddingPlane", tables: Sequence[InputTable], target: Owner, stack: ExitStack, *, gather: bool
) -> None:
	for table in tables:
		if table.values is None:
			plane.supply_inputs(table.chunks, target, table.width, stack, gather=gather)
