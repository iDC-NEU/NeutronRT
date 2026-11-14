"""Storage-compiled task packets for GPU-owned full-neighbor execution."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import TensorSpec
from lib.graph.ownership import ComputeAttempt, Location
from lib.graph.recompute import LayerPlan

WIRE_VERSION = 2
COMPUTE = (WIRE_VERSION << 8) | 1
READ = (WIRE_VERSION << 8) | 2
RETIRE = (WIRE_VERSION << 8) | 3
STOP = (WIRE_VERSION << 8) | 4
READY = (WIRE_VERSION << 8) | 5
ERROR = (WIRE_VERSION << 8) | 6
PROFILE = (WIRE_VERSION << 8) | 7
READ_MANY = (WIRE_VERSION << 8) | 8
PREPARE = (WIRE_VERSION << 8) | 9
PREPARED = (WIRE_VERSION << 8) | 10
RUN_PREPARED = (WIRE_VERSION << 8) | 11
CANCEL_PREPARED = (WIRE_VERSION << 8) | 12
COMPUTE_WITH_AGGREGATE = (WIRE_VERSION << 8) | 13
RECONSTRUCT = (WIRE_VERSION << 8) | 14
COMPUTE_FROM_AGGREGATES = (WIRE_VERSION << 8) | 15
DELTA_COMPUTE = (WIRE_VERSION << 8) | 16


@dataclass(frozen=True)
class InputChunk:
	location: Location
	rows: Tensor
	positions: Tensor
	sender: Optional[int] = None


@dataclass(frozen=True)
class ComputeHeader:
	request: int
	generation: int
	layer: int
	sources: int
	targets: int
	edges: int
	groups: int
	has_degrees: bool = False
	edge_width: int = 0

	def tensor(self, *, opcode: int = COMPUTE) -> Tensor:
		if opcode not in (COMPUTE, PREPARE, COMPUTE_WITH_AGGREGATE):
			raise ValueError("Unsupported task command")
		return torch.tensor(
			[
				opcode,
				self.request,
				self.generation,
				self.layer,
				self.sources,
				self.targets,
				self.edges,
				(self.edge_width << 32) | (self.groups << 1) | int(self.has_degrees),
			],
			dtype=torch.int64,
		)

	@classmethod
	def decode(cls, value: Tensor, *, opcode: int = COMPUTE) -> "ComputeHeader":
		if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != (8,):
			raise ValueError("Owned control headers require eight CPU int64 fields")
		fields = value.tolist()
		if opcode not in (COMPUTE, PREPARE, COMPUTE_WITH_AGGREGATE) or fields[0] != opcode:
			raise ValueError("Unsupported owned compute opcode or wire version")
		header = cls(
			fields[1],
			fields[2],
			fields[3],
			fields[4],
			fields[5],
			fields[6],
			(fields[7] & 0xFFFFFFFF) >> 1,
			bool(fields[7] & 1),
			fields[7] >> 32,
		)
		if min(header.request, header.generation, header.layer) < 1:
			raise ValueError("Invalid owned compute identity")
		if min(header.sources, header.targets, header.edges, header.groups, header.edge_width) < 0:
			raise ValueError("Negative owned task dimensions")
		if header.layer == 1 and header.groups or header.layer > 1 and header.sources > 0 and not header.groups:
			raise ValueError("Owned task inputs do not match the layer")
		return header

	def specs(self, width: int) -> Tuple[TensorSpec, ...]:
		if width < 1:
			raise ValueError("Input width must be positive")
		return (
			TensorSpec((3,), torch.int64),
			TensorSpec((self.edges,), torch.int64),
			TensorSpec((self.edges,), torch.int64),
			TensorSpec((self.sources,), torch.int64),
			TensorSpec((self.targets,), torch.int64),
			TensorSpec((self.sources if self.has_degrees else 0,), torch.int64),
			TensorSpec((self.groups, 4), torch.int64),
			TensorSpec((self.sources if self.groups else 0,), torch.int64),
			TensorSpec((self.sources if self.groups else 0,), torch.int64),
			TensorSpec((self.sources if self.layer == 1 else 0, width), torch.float32),
		) + ((TensorSpec((self.edges, self.edge_width), torch.float32),) if self.edge_width else ())


@dataclass(frozen=True)
class ComputePacket:
	header: ComputeHeader
	fields: Tuple[Tensor, ...]


def _ids(value: Tensor, count: int) -> None:
	if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != (count,):
		raise ValueError("Task node and row metadata must be CPU int64 vectors")


def pack_compute(
	task: LayerPlan,
	attempt: ComputeAttempt,
	*,
	width: int,
	inputs: Sequence[InputChunk] = (),
	features: Optional[Tensor] = None,
) -> ComputePacket:
	"""Retain graph/features by reference; allocate only compact routing metadata."""
	if task.block.device.type != "cpu" or task.layer != attempt.key.layer or attempt.key.kind != "h":
		raise ValueError("Owned tasks must come from the storage CPU at the attempt's layer")
	header = ComputeHeader(
		attempt.number,
		attempt.owner.generation,
		task.layer,
		task.sources.numel(),
		task.targets.numel(),
		task.block.num_edges(),
		len(inputs),
		task.source_out_degrees is not None,
		task.block.edata["nrt"].shape[1] if "nrt" in task.block.edata else 0,
	)
	ComputeHeader.decode(header.tensor())
	_ids(task.sources, header.sources)
	_ids(task.targets, header.targets)
	if task.block.num_src_nodes() != header.sources or task.block.num_dst_nodes() != header.targets:
		raise ValueError("Task row IDs do not match the compiled block")
	if task.layer == 1:
		if features is None or features.shape != (header.sources, width):
			raise ValueError("First-layer features do not match the compiled sources")
		if features.device.type != "cpu" or features.dtype != torch.float32:
			raise ValueError("Owned feature payloads must be CPU float32")
	elif features is not None:
		raise ValueError("Intermediate inputs must be resolved through their owners")
	for item in inputs:
		_ids(item.rows, item.rows.numel())
		_ids(item.positions, item.rows.numel())
		if item.rows.numel() and item.rows.min() < 0:
			raise ValueError("Input chunk offsets must be nonnegative")
		if item.location.handle < 0:
			raise ValueError("Input handles must be nonnegative")
		if item.location.owner.rank < 0 or item.location.owner.generation < 1:
			raise ValueError("Invalid input owner identity")
		if item.sender is not None and (
			type(item.sender) is not int or item.sender != 0 or item.location.owner == attempt.owner
		):
			raise ValueError("An input proxy must use the storage coordinator for a remote source")
	rows = torch.cat([item.rows for item in inputs]) if inputs else torch.empty(0, dtype=torch.int64)
	positions = torch.cat([item.positions for item in inputs]) if inputs else torch.empty(0, dtype=torch.int64)
	if inputs and not torch.equal(positions.sort().values, torch.arange(header.sources)):
		raise ValueError("Input positions must partition all compiled source rows")
	metadata = torch.tensor(
		[
			[
				item.location.owner.rank if item.sender is None else item.sender,
				item.location.owner.generation,
				item.location.handle,
				item.rows.numel(),
			]
			for item in inputs
		],
		dtype=torch.int64,
	).reshape(-1, 4)
	degrees = task.source_out_degrees
	if degrees is not None:
		_ids(degrees, header.sources)
		if degrees.numel() and degrees.min() < 0:
			raise ValueError("Source degrees must be nonnegative")
	u, v = task.block.edges()
	fields = (
		torch.tensor([attempt.key.version, attempt.key.chunk, int(degrees is not None)], dtype=torch.int64),
		u,
		v,
		task.sources,
		task.targets,
		degrees if degrees is not None else torch.empty(0, dtype=torch.int64),
		metadata,
		rows,
		positions,
		features if features is not None else torch.empty(0, width, dtype=torch.float32),
	) + ((task.block.edata["nrt"],) if header.edge_width else ())
	if tuple(TensorSpec.of(value) for value in fields) != header.specs(width):
		raise ValueError("Compiled fields do not match the owned wire schema")
	return ComputePacket(header, fields)
