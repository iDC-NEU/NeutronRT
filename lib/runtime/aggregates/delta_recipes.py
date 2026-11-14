"""Selected old/current worker inputs for sparse aggregate-only inference."""

from contextlib import ExitStack
from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor

from lib.graph.delta_plan import SparseLayerPlan
from lib.graph.ownership import Owner
from lib.runtime.aggregates.delta_inputs import DeltaInputs
from lib.runtime.aggregates.delta_state import DeltaContext
from lib.runtime.aggregates.history import HistoricalAggregates
from lib.runtime.aggregates.tables import InputTable
from lib.runtime.aggregates.temporary import ReconstructionWorkspace, TemporaryReconstruction
from lib.runtime.protocol import InputChunk

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


def _select(table: InputTable, positions: Tensor) -> InputTable:
	"""Select unique compact rows using integer routes, without fetching values."""
	remap = torch.full((table.rows,), -1, dtype=torch.int64)
	remap[positions] = torch.arange(positions.numel())
	chunks = []
	for chunk in table.chunks:
		selected = remap[chunk.positions] >= 0
		if selected.any():
			chunks.append(
				InputChunk(chunk.location, chunk.rows[selected], remap[chunk.positions[selected]], chunk.sender)
			)
	return InputTable(positions.numel(), table.width, chunks=tuple(chunks))


class AggregateDeltaRecipes:
	def __init__(
		self, service: "DistributedEmbeddingService", context: DeltaContext, layer: int, plan: SparseLayerPlan
	):
		assert service._aggregate_only is not None and layer > 1
		self.service, self.context, self.layer, self.plan = service, context, layer - 1, plan
		self.current = torch.cat((plan.targets, plan.inputs.value_nodes)).unique(sorted=True)
		self.old = torch.cat((plan.inputs.value_nodes, plan.edges.value_nodes)).unique(sorted=True)
		self.history = HistoricalAggregates(service)
		self.temporary = TemporaryReconstruction(service)

	@property
	def workspace_bytes(self) -> int:
		# Current h may be evicted during admission. Budget its worst-case misses.
		return int((self.current.numel() + self.old.numel()) * self.service.widths[self.layer] * 4)

	def prepare(self) -> None:
		assert self.service._aggregate_only is not None
		self.history.prepare(self.context, self.old, self.layer)
		self.service._aggregate_only.ensure_prefix(self.current, self.layer)

	def tables(self, target: Owner, stack: ExitStack, workspace: ReconstructionWorkspace) -> Tuple[InputTable, ...]:
		service, nodes, layer = self.service, self.current, self.layer
		assert service._aggregate_only is not None
		# This runs after the single output+workspace admission. No make_room or
		# other model task can retire these private h buffers until delta drains.
		cached = service.catalog.available(layer, nodes, service.catalog.version)
		current = service._aggregate_only.table(service.catalog, layer, nodes[cached], target)
		chunks = [
			InputChunk(chunk.location, chunk.rows, cached.nonzero().flatten()[chunk.positions], chunk.sender)
			for chunk in current.chunks
		]
		missing = (~cached).nonzero().flatten()
		if missing.numel():
			reconstructed = self.temporary.create(
				nodes[missing],
				layer,
				service._aggregate_only.tables(nodes[missing], layer, target),
				service.graph.in_degrees(nodes[missing]),
				target,
				service.catalog.version,
				stack,
				workspace=workspace,
			)
			chunks.extend(
				InputChunk(chunk.location, chunk.rows, missing[chunk.positions]) for chunk in reconstructed.chunks
			)
		current = InputTable(nodes.numel(), service.widths[layer], chunks=tuple(chunks))
		old = self.temporary.create(
			self.old,
			layer,
			self.history.tables(self.context, self.old, layer, target),
			self.history.degrees(self.context, self.old, layer),
			target,
			self.context.version - 1,
			stack,
			workspace=workspace,
		)
		plan = self.plan
		return (
			DeltaInputs(service).old_sum(self.context.layers[layer + 1].aggregates, plan.targets, target),
			_select(current, torch.searchsorted(nodes, plan.targets)),
			_select(current, torch.searchsorted(nodes, plan.inputs.value_nodes)),
			_select(old, torch.searchsorted(self.old, plan.inputs.value_nodes)),
			_select(old, torch.searchsorted(self.old, plan.edges.value_nodes)),
		)
