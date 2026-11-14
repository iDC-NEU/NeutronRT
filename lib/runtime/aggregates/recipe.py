"""Reconstruct ephemeral source embeddings before a storage-compiled SAGE block."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt
from lib.graph.recompute import LayerPlan
from lib.runtime.aggregates.reconstruction import ReconstructionHeader, validate_recipe
from lib.runtime.aggregates.tables import InputTable, pack_tables
from lib.runtime.protocol import COMPUTE_FROM_AGGREGATES


@dataclass(frozen=True)
class RecipeHeader:
	request: int
	generation: int
	layer: int
	sources: int
	targets: int
	edges: int
	nbytes: int

	def tensor(self) -> Tensor:
		return torch.tensor(
			[
				COMPUTE_FROM_AGGREGATES,
				self.request,
				self.generation,
				self.layer,
				self.sources,
				self.targets,
				self.edges,
				self.nbytes,
			],
			dtype=torch.int64,
		)

	@classmethod
	def decode(cls, value: Tensor) -> "RecipeHeader":
		if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != (8,):
			raise ValueError("Aggregate compute recipes need eight CPU int64 fields")
		opcode, *fields = value.tolist()
		header = cls(*fields)
		if (
			opcode != COMPUTE_FROM_AGGREGATES
			or min(header.request, header.generation) < 1
			or header.layer < 2
			or min(header.sources, header.targets, header.edges) < 0
			or header.targets > header.sources
			or header.nbytes < header.layer * 32 + 32 + (header.edges * 2 + header.sources * 2 + header.targets) * 8
		):
			raise ValueError("Invalid aggregate compute recipe identity or dimensions")
		return header

	def reconstruction(self) -> ReconstructionHeader:
		return ReconstructionHeader(self.request, self.generation, self.layer - 1, 0, self.sources, self.nbytes)

	def tail_specs(self) -> Tuple[TensorSpec, ...]:
		return (
			TensorSpec((4,), torch.int64),
			TensorSpec((self.edges,), torch.int64),
			TensorSpec((self.edges,), torch.int64),
			TensorSpec((self.sources,), torch.int64),
			TensorSpec((self.targets,), torch.int64),
			TensorSpec((self.sources,), torch.int64),
		)


@dataclass(frozen=True)
class RecipePacket:
	header: RecipeHeader
	fields: Tuple[Tensor, ...]


def pack_recipe_compute(
	task: LayerPlan,
	embedding: ComputeAttempt,
	aggregate: ComputeAttempt,
	widths: Sequence[int],
	tables: Sequence[InputTable],
	degrees: Tensor,
) -> RecipePacket:
	if (
		embedding.key.kind != "h"
		or aggregate.key.kind != "a"
		or embedding.owner != aggregate.owner
		or embedding.key.version != aggregate.key.version
		or embedding.key.layer != aggregate.key.layer
		or task.layer != embedding.key.layer
		or min(embedding.number, aggregate.number) < 1
		or embedding.number == aggregate.number
		or task.block.device.type != "cpu"
	):
		raise ValueError("Aggregate recipe outputs need distinct same-owner/version/layer attempts")
	inputs = pack_tables(tables)
	if (
		degrees.device.type != "cpu"
		or degrees.dtype != torch.int64
		or degrees.shape != task.sources.shape
		or (degrees < 0).any()
	):
		raise ValueError("Aggregate recipe degrees must align nonnegative source rows")
	if any(
		value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1
		for value in (task.sources, task.targets)
	):
		raise ValueError("Aggregate recipe source and target IDs require CPU int64 vectors")
	if task.block.num_src_nodes() != task.sources.numel() or task.block.num_dst_nodes() != task.targets.numel():
		raise ValueError("Aggregate recipe block dimensions do not match node tables")
	u, v = task.block.edges()
	fields = (
		*inputs.fields,
		torch.tensor(
			[embedding.key.version, embedding.key.chunk, aggregate.number, aggregate.key.chunk], dtype=torch.int64
		),
		u,
		v,
		task.sources,
		task.targets,
		degrees,
	)
	header = RecipeHeader(
		embedding.number,
		embedding.owner.generation,
		task.layer,
		task.sources.numel(),
		task.targets.numel(),
		task.block.num_edges(),
		PackedLayout(tuple(TensorSpec.of(field) for field in fields)).nbytes,
	)
	RecipeHeader.decode(header.tensor())
	validate_recipe(header.reconstruction(), inputs.fields, widths)
	return RecipePacket(header, fields)
