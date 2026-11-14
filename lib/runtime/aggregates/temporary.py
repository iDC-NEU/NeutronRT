"""Worker-local reconstructed inputs whose keys never enter the row catalog."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, Sequence

import torch
from torch import Tensor

from lib.graph.ownership import Owner
from lib.runtime.aggregates.reconstruction import pack_reconstruction
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.protocol import InputChunk

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


@dataclass
class ReconstructionWorkspace:
	service: "DistributedEmbeddingService"
	owner: Owner
	remaining: int
	active: bool = True


class TemporaryReconstruction:
	def __init__(self, service: "DistributedEmbeddingService") -> None:
		self.service = service

	@contextmanager
	def workspace(self, target: Owner, required: int, *, output_bytes: int = 0) -> Iterator[ReconstructionWorkspace]:
		if type(required) is not int or type(output_bytes) is not int or not 0 <= output_bytes <= required:
			raise ValueError("Reconstruction workspace requires nonnegative byte counts with reserved outputs")
		self.service.remote_plane.make_room(target, required)
		admission = ReconstructionWorkspace(self.service, target, required - output_bytes)
		try:
			yield admission
		finally:
			admission.active = False

	def create(
		self,
		nodes: Tensor,
		layer: int,
		tables: Sequence[InputTable],
		degrees: Tensor,
		target: Owner,
		version: int,
		stack: ExitStack,
		*,
		workspace: ReconstructionWorkspace,
	) -> InputTable:
		"""Caller admits all workspace once; this operation must not evict inputs."""
		service, plane = self.service, self.service.remote_plane
		if type(version) is not int or not 0 <= version <= service.catalog.version:
			raise ValueError("Temporary reconstruction version is outside graph history")
		if not 1 <= layer < len(service.widths):
			raise ValueError("Temporary reconstruction layer is outside the model")
		if (
			nodes.device.type != "cpu"
			or nodes.dtype != torch.int64
			or nodes.ndim != 1
			or nodes.unique().numel() != nodes.numel()
			or degrees.shape != nodes.shape
		):
			raise ValueError("Temporary reconstruction needs unique CPU node IDs and aligned degrees")
		width = service.widths[layer]
		if not nodes.numel():
			return InputTable(0, width)
		if not workspace.active or workspace.service is not service or workspace.owner != target:
			raise ValueError("Temporary reconstruction requires its live worker workspace")
		requested = nodes.numel() * width * 4
		if requested > workspace.remaining:
			raise MemoryError("Temporary reconstruction workspace was not admitted")
		# Ready replies can still count retired buffers with transfer leases. The
		# worker drains those leases before retrying allocation; never re-admit
		# against that transient report between already-reserved temporaries.
		workspace.remaining -= requested
		# Consume the catalog's global chunk sequence without publishing any rows.
		reserved = service.catalog.reserve(layer, nodes, service.catalog.version)
		service.catalog.cancel(reserved)
		key = replace(reserved, version=version)
		stack.callback(plane.retire, key)
		attempt = plane.directory.begin(key, target)
		packet = pack_reconstruction(attempt, service.widths, tables, degrees)
		with ExitStack() as inputs:
			result = plane.transport.reconstruct(
				target,
				packet,
				output_width=width,
				supply_inputs=lambda: supply_tables(plane, tables, target, inputs, gather=service.gather_inputs),
			)
		location = plane.publish_result(attempt, result)
		service.counters.reconstruction_tasks += 1
		service.counters.reconstructed_rows += nodes.numel()
		positions = torch.arange(nodes.numel())
		return InputTable(nodes.numel(), width, chunks=(InputChunk(location, positions, positions),))
