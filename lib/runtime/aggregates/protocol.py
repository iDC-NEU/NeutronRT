"""Paired output packets reuse existing input routing and add aggregate identity."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import TensorSpec
from lib.graph.ownership import ComputeAttempt
from lib.graph.recompute import LayerPlan
from lib.runtime.protocol import ComputeHeader, ComputePacket, InputChunk, pack_compute


def aggregate_specs(header: ComputeHeader, width: int) -> Tuple[TensorSpec, ...]:
	return (*header.specs(width), TensorSpec((4,), torch.int64))


def aggregate_identity(header: ComputeHeader, fields: Tuple[Tensor, ...]) -> tuple[ComputeHeader, Tensor]:
	"""Validate the independent aggregate attempt after draining the task body."""
	if len(fields) != 11 or fields[-1].shape != (4,) or fields[-1].dtype != torch.int64:
		raise ValueError("Aggregate identity requires four int64 fields after normal compute inputs")
	request, version, layer, chunk = fields[-1].cpu().tolist()
	if (
		min(request, layer) < 1
		or min(version, chunk) < 0
		or request == header.request
		or layer != header.layer
		or version != int(fields[0][0])
	):
		raise ValueError("Aggregate identity must name a distinct same-version, same-layer attempt")
	metadata = torch.tensor([version, chunk, int(header.has_degrees)], dtype=torch.int64)
	return replace(header, request=request), metadata


@dataclass(frozen=True)
class AggregatePacket:
	embedding: ComputePacket
	identity: Tensor

	@property
	def header(self) -> ComputeHeader:
		return self.embedding.header

	@property
	def fields(self) -> Tuple[Tensor, ...]:
		return (*self.embedding.fields, self.identity)

	def result_packets(self) -> tuple[ComputePacket, ComputePacket]:
		"""Match each completion against its own attempt; chunk sequences may differ."""
		header, metadata = aggregate_identity(self.header, self.fields)
		return self.embedding, ComputePacket(header, (metadata, *self.embedding.fields[1:]))


def pack_aggregate_compute(
	task: LayerPlan,
	embedding: ComputeAttempt,
	aggregate: ComputeAttempt,
	*,
	width: int,
	inputs: Sequence[InputChunk] = (),
	features: Optional[Tensor] = None,
) -> AggregatePacket:
	if (
		aggregate.key.kind != "a"
		or aggregate.owner != embedding.owner
		or aggregate.key.version != embedding.key.version
		or aggregate.key.layer != embedding.key.layer
		or aggregate.number < 1
		or aggregate.number == embedding.number
	):
		raise ValueError("Paired aggregate and embedding attempts need one owner/version/layer and distinct requests")
	packet = AggregatePacket(
		pack_compute(task, embedding, width=width, inputs=inputs, features=features),
		torch.tensor(
			[aggregate.number, aggregate.key.version, aggregate.key.layer, aggregate.key.chunk], dtype=torch.int64
		),
	)
	packet.result_packets()
	return packet
