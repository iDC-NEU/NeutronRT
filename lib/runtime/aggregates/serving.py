"""Keep backed sums as derived state and h as a bounded reconstructible cache."""

from contextlib import ExitStack
from typing import TYPE_CHECKING, Sequence, Tuple

import torch
from torch import Tensor

from lib.graph.catalog import EmbeddingCatalog
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.graph.recompute import LayerPlan
from lib.graph.serving import select_rows
from lib.runtime.aggregates.recipe import pack_recipe_compute
from lib.runtime.aggregates.reconstruction import pack_reconstruction
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.protocol import InputChunk
from lib.runtime.transport import ReadyResult

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


class AggregateOnly:
	def __init__(self, service: "DistributedEmbeddingService") -> None:
		self.service = service
		assert service.aggregate_catalog is not None
		self.catalog = service.aggregate_catalog
		self.plane = service.remote_plane
		self.plane.discard_embeddings = True

	def ensure_prefix(self, nodes: Tensor, layer: int) -> None:
		for prefix in range(1, layer + 1):
			self.service._ensure_rows(nodes, prefix)

	def table(self, catalog: EmbeddingCatalog, layer: int, nodes: Tensor, target: Owner) -> InputTable:
		chunks = []
		for group in catalog.selections(layer, nodes, self.service.catalog.version):
			location = self.plane.directory.locate(group.key, preferred=target)
			proxy = not self.service.peer_fetch and location.owner not in (target, self.plane.storage.owner)
			chunks.append(InputChunk(location, group.rows, group.positions, sender=0 if proxy else None))
		return InputTable(nodes.numel(), self.service.widths[layer], chunks=tuple(chunks))

	def tables(self, nodes: Tensor, layer: int, target: Owner, *, start: int = 0) -> Tuple[InputTable, ...]:
		first = (
			self.table(self.service.catalog, start, nodes, target)
			if start
			else InputTable(nodes.numel(), self.service.widths[0], values=select_rows(self.service.features, nodes))
		)
		return (first, *(self.table(self.catalog, prefix, nodes, target) for prefix in range(start + 1, layer + 1)))

	def supply(self, tables: Sequence[InputTable], target: Owner, stack: ExitStack) -> None:
		supply_tables(self.plane, tables, target, stack, gather=self.service.gather_inputs)

	def compute(
		self, task: LayerPlan, h: ComputeAttempt, a: ComputeAttempt, stack: ExitStack
	) -> Tuple[ReadyResult, ReadyResult]:
		tables = self.tables(task.sources, task.layer - 1, h.owner)
		packet = pack_recipe_compute(
			task, h, a, self.service.widths, tables, self.service.graph.in_degrees(task.sources)
		)
		return self.plane.transport.compute_from_aggregates(
			h.owner,
			packet,
			output_width=self.service.widths[task.layer],
			supply_inputs=lambda: self.supply(tables, h.owner, stack),
		)

	def reconstruct(self, nodes: Tensor, layer: int, stack: ExitStack) -> Tensor:
		service, plane = self.service, self.plane
		self.ensure_prefix(nodes, layer)
		worker = service._pick(required=nodes, input_layer=layer - 1, task_cost=max(1, nodes.numel()))
		key = None
		published = False
		obsolete: Tuple[StateKey, ...] = ()
		try:
			plane.make_room(worker.owner, nodes.numel() * service.widths[layer] * 4)
			start = next(
				(
					prefix
					for prefix in range(layer - 1, 0, -1)
					if service.catalog.available(prefix, nodes, service.catalog.version).all()
				),
				0,
			)
			tables = self.tables(nodes, layer, worker.owner, start=start)
			key = service.catalog.reserve(layer, nodes, service.catalog.version)
			attempt = plane.directory.begin(key, worker.owner)
			packet = pack_reconstruction(
				attempt, service.widths, tables, service.graph.in_degrees(nodes), start_layer=start
			)
			with ExitStack() as inputs:
				result = plane.transport.reconstruct(
					worker.owner,
					packet,
					output_width=service.widths[layer],
					supply_inputs=lambda: self.supply(tables, worker.owner, inputs),
				)
			plane.publish_result(attempt, result)
			obsolete = service.catalog.publish(key)
			published = True
			service.counters.reconstruction_tasks += 1
			service.counters.reconstructed_rows += nodes.numel()
			return plane.read_rows(key, torch.arange(nodes.numel()), stack)
		finally:
			if key is not None and not published:
				service.catalog.cancel(key)
				plane.retire(key)
			for retired in obsolete:
				plane.retire(retired)
			service._release_dispatch(worker)

	def load(self, layer: int, nodes: Tensor, owner: Owner, stack: ExitStack) -> Tensor:
		service = self.service
		if service.catalog.available(layer, nodes, service.catalog.version).all():
			return service._load_embeddings(layer, nodes, owner, stack)
		output = torch.empty(nodes.numel(), service.widths[layer], dtype=torch.float32)
		for positions in torch.arange(nodes.numel()).split(service.batch_size):
			selected = nodes[positions]
			cached = service.catalog.available(layer, selected, service.catalog.version)
			if cached.any():
				output[positions[cached]] = service._load_embeddings(layer, selected[cached], owner, stack)
			if not cached.all():
				# Read cached rows before make_room can discard them. Publish only unique
				# misses, then restore duplicate query positions from the returned rows.
				missing, inverse = selected[~cached].unique(sorted=True, return_inverse=True)
				output[positions[~cached]] = self.reconstruct(missing, layer, stack)[inverse]
		return output
