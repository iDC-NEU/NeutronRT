"""Run the shared versioned update/query pipeline against process-owned workers."""

from contextlib import ExitStack
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Optional, Sequence, Tuple, TypedDict

import torch
from dgl import DGLGraph
from torch import Tensor

from lib.graph.affected import AffectedLayer
from lib.graph.catalog import EmbeddingCatalog, RowSelection
from lib.graph.delta_plan import SparseLayerPlan
from lib.graph.neighborhood import split_task
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.graph.recompute import LayerPlan, RecomputePlan
from lib.graph.serving import AppliedMutation, EmbeddingService, UpdateResult, _PreparedLayer, select_rows
from lib.runtime.aggregates.serving import AggregateOnly
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk, pack_compute
from lib.runtime.transport import ReadyResult, RemoteEndpoint


def _remote_only(_worker: RemoteEndpoint, _task: LayerPlan, _inputs: Tensor) -> Tensor:
	raise RuntimeError("Remote execution uses resident handles rather than coordinator GPU tensors")


@dataclass
class _Reservation:
	catalog: EmbeddingCatalog
	key: StateKey
	attempt: Optional[ComputeAttempt] = None
	published: bool = False


class _ComputeArguments(TypedDict):
	width: int
	inputs: Sequence[InputChunk]
	features: Optional[Tensor]


class DistributedEmbeddingService(EmbeddingService[RemoteEndpoint]):
	"""Keep storage analysis/admission unchanged; replace only task execution/load.

	The coordinator never acquires a CUDA context. Compute inputs route from their
	current CPU/GPU owners to the target worker. Frontend queries gather selected
	CPU rows for consumption/serialization, without creating an eager full replica.
	"""

	def __init__(
		self,
		graph: DGLGraph,
		features: Tensor,
		widths: Sequence[int],
		plane: RemoteEmbeddingPlane,
		*,
		peer_fetch: bool = True,
		gather_inputs: bool = False,
		concurrent_compute: bool = False,
		retain_aggregates: bool = False,
		aggregate_only: bool = False,
		incremental_aggregates: bool = False,
		row_refresh: bool = False,
		delta_max_steps: int = 32,
		delta_anchor_bytes: int = 256 * 1024**2,
		**kwargs,
	) -> None:
		if features.dtype != torch.float32:
			raise ValueError("Process-owned model execution currently requires float32 features")
		if retain_aggregates and concurrent_compute:
			raise NotImplementedError("Retained aggregates require serial dispatch until paired wave integration")
		if aggregate_only and not retain_aggregates:
			raise ValueError("Aggregate-only storage requires retained projected sums")
		if incremental_aggregates and not retain_aggregates:
			raise ValueError("Sparse updates require retained aggregates")
		if row_refresh and not incremental_aggregates:
			raise ValueError("Row refresh requires incremental_aggregates=true")
		super().__init__(graph, features, widths, plane, list(plane.remotes.values()), _remote_only, **kwargs)
		self.remote_plane = plane
		self.peer_fetch = peer_fetch
		self.gather_inputs = gather_inputs
		self.concurrent_compute = concurrent_compute
		self.retain_aggregates = retain_aggregates
		self.aggregate_catalog: Optional[EmbeddingCatalog] = None
		if retain_aggregates:
			self.aggregate_catalog = EmbeddingCatalog(graph.num_nodes(), len(widths) - 1, plane.directory, kind="a")
			self._additional_catalogs = (self.aggregate_catalog,)
		self.aggregate_only = aggregate_only
		self._aggregate_only = AggregateOnly(self) if aggregate_only else None
		from lib.runtime.aggregates.delta_serving import DeltaExecution
		from lib.runtime.aggregates.delta_state import DeltaState

		self._delta = (
			DeltaExecution(
				DeltaState(self, max_steps=delta_max_steps, anchor_bytes=delta_anchor_bytes, row_refresh=row_refresh)
			)
			if incremental_aggregates
			else None
		)

	def _mutate_features(self, nodes: Tensor, values: Tensor) -> AppliedMutation:
		start = perf_counter()
		try:
			if self._delta is not None:
				self._delta.state.prepare_features(nodes)
			mutation = super()._mutate_features(nodes, values)
			return replace(mutation, start=start)
		except BaseException:
			if self._delta is not None:
				self._delta.state.pending = None
			raise

	def _mutate_edges(
		self, sources: Tensor, targets: Tensor, *, delete: bool, attributes: Optional[Tensor] = None
	) -> AppliedMutation:
		start = perf_counter()
		try:
			if self._delta is not None:
				self._delta.state.prepare_edges(sources, targets, delete=delete)
			mutation = super()._mutate_edges(sources, targets, delete=delete, attributes=attributes)
			return replace(mutation, start=start)
		except BaseException:
			if self._delta is not None:
				self._delta.state.pending = None
			raise

	def _finish_mutation(self, mutation: AppliedMutation) -> UpdateResult:
		if self._delta is None:
			return super()._finish_mutation(mutation)
		with self._delta.state.updating():
			return super()._finish_mutation(mutation)

	def _finish_analysis(self, layer: int, dirty: Tensor, version: int) -> None:
		if self._delta is not None and self._delta.state.active is not None:
			self._delta.state.active.capture(layer, dirty, version)
		super()._finish_analysis(layer, dirty, version)

	def _prepare_layer(self, affected: AffectedLayer) -> _PreparedLayer:
		if self._delta is not None and self._delta.state.active is not None:
			return _PreparedLayer(affected, None)
		return super()._prepare_layer(affected)

	def _run_delta(self, layer: int, plan: SparseLayerPlan) -> None:
		assert self._delta is not None
		self._delta.run(layer, plan)

	def _state_available(self, layer: int, nodes: Tensor, version: int) -> Tensor:
		if self._aggregate_only is not None:
			return self._aggregate_only.catalog.available(layer, nodes, version)
		return super()._state_available(layer, nodes, version)

	def _run_wave(self, tasks: Sequence[LayerPlan]) -> None:
		from lib.runtime.dispatch import run_wave

		wave: list[LayerPlan] = []
		for task in tasks:
			for block in self._split_task(task):
				wave.append(block)
				if len(wave) == len(self.workers):
					run_wave(self, wave)
					wave = []
		if wave:
			run_wave(self, wave)

	def _compute_layer(self, prepared: _PreparedLayer) -> None:
		if self._delta is not None and self._delta.state.active is not None:
			return self._delta.compute(prepared)
		if not self.concurrent_compute or len(self.workers) < 2:
			return super()._compute_layer(prepared)
		layer, targets = prepared.affected.layer, prepared.affected.targets
		wave: list[LayerPlan] = []
		for index, chunk in enumerate(targets.split(self.batch_size)):
			if self.catalog.available(layer, chunk, self.catalog.version).all():
				continue
			task = prepared.first if index == 0 else None
			tasks = (task,) if task is not None else self.planner._blocks(layer, chunk, self.catalog.version)
			for compiled in tasks:
				for block in self._split_task(compiled):
					wave.append(block)
					if len(wave) == len(self.workers):
						self._run_wave(wave)
						self._yield_compute()
						wave = []
		if wave:
			self._run_wave(wave)
			self._yield_compute()

	def _execute_plan(self, plan: RecomputePlan) -> None:
		for task in plan.tasks:
			if self.concurrent_compute and len(self.workers) > 1:
				self._run_wave((task,))
			else:
				for block in self._split_task(task):
					self._run_task(block, plan.version)

	def _split_task(self, task: LayerPlan):
		# Keep normal/aggregate task bodies safely below the wire allocation guard.
		return split_task(task, self.batch_size, input_width=self.widths[task.layer - 1], max_bytes=512 * 1024**2)

	@torch.inference_mode()
	def _run_task(self, task: LayerPlan, version: int) -> None:
		plane = self.remote_plane
		if task.layer > 1:
			self._ensure_rows(task.sources, task.layer - 1)
			if self._aggregate_only is not None:
				self._aggregate_only.ensure_prefix(task.sources, task.layer - 1)
		task_cost = self._task_cost(task)
		worker = self._pick(required=task.sources, input_layer=task.layer - 1, task_cost=task_cost)
		outputs = []
		obsolete: list[StateKey] = []
		try:
			plane.make_room(
				worker.owner, task.targets.numel() * self.widths[task.layer] * 4 * len(self._state_catalogs())
			)
			for catalog in self._state_catalogs():
				key = catalog.reserve(task.layer, task.targets, version)
				output = _Reservation(catalog, key)
				outputs.append(output)
				output.attempt = plane.directory.begin(key, worker.owner)
			recipe = (
				self._aggregate_only is not None
				and task.layer > 1
				and not self.catalog.available(task.layer - 1, task.sources, version).all()
			)
			groups = (
				self.catalog.selections(task.layer - 1, task.sources, version) if task.layer > 1 and not recipe else ()
			)
			inputs = []
			for group in groups:
				location = plane.directory.locate(group.key, preferred=worker.owner)
				proxy = not self.peer_fetch and location.owner not in (worker.owner, plane.storage.owner)
				inputs.append(InputChunk(location, group.rows, group.positions, sender=0 if proxy else None))
			with ExitStack() as stack:

				def supply_inputs(inputs=inputs, worker=worker, task=task, stack=stack):
					plane.supply_inputs(
						inputs, worker.owner, self.widths[task.layer - 1], stack, gather=self.gather_inputs
					)

				arguments: _ComputeArguments = {
					"width": self.widths[task.layer - 1],
					"inputs": inputs,
					"features": select_rows(self.features, task.sources) if task.layer == 1 else None,
				}
				assert outputs[0].attempt is not None
				results: Tuple[ReadyResult, ...]
				if self.retain_aggregates:
					from lib.runtime.aggregates.protocol import pack_aggregate_compute

					assert outputs[1].attempt is not None
					if recipe:
						assert self._aggregate_only is not None
						results = self._aggregate_only.compute(task, outputs[0].attempt, outputs[1].attempt, stack)
					else:
						paired = pack_aggregate_compute(task, outputs[0].attempt, outputs[1].attempt, **arguments)
						results = plane.transport.compute_with_aggregate(
							worker.owner, paired, output_width=self.widths[task.layer], supply_inputs=supply_inputs
						)
				else:
					packet = pack_compute(task, outputs[0].attempt, **arguments)
					results = (
						plane.transport.compute(
							worker.owner, packet, output_width=self.widths[task.layer], supply_inputs=supply_inputs
						),
					)
				if len(results) != len(outputs):
					raise RuntimeError("Worker result count does not match the reserved derived outputs")
				for output, result in zip(outputs, results, strict=True):
					assert output.attempt is not None
					plane.publish_result(output.attempt, result)
				for output in outputs:
					obsolete.extend(output.catalog.publish(output.key))
					output.published = True
				if self._delta is not None:
					self._delta.state.record(outputs[1].key, 0, obsolete)
				self.counters.tasks += 1
				self.counters.rows += task.targets.numel()
		finally:
			for output in outputs:
				if not output.published:
					output.catalog.cancel(output.key)
					plane.retire(output.key)
			for retired in obsolete:
				plane.retire(retired)
			self._release_dispatch(worker)

	def _load(
		self,
		layer: int,
		nodes: Tensor,
		owner: Owner,
		stack: ExitStack,
		selections: Optional[Tuple[RowSelection, ...]] = None,
	) -> Tensor:
		if self._aggregate_only is not None and layer > 0 and selections is None:
			if owner != self.remote_plane.storage.owner:
				raise ValueError("Frontend reads target the storage CPU")
			return self._aggregate_only.load(layer, nodes, owner, stack)
		return self._load_embeddings(layer, nodes, owner, stack, selections)

	def _load_embeddings(
		self,
		layer: int,
		nodes: Tensor,
		owner: Owner,
		stack: ExitStack,
		selections: Optional[Tuple[RowSelection, ...]] = None,
	) -> Tensor:
		plane = self.remote_plane
		if owner != plane.storage.owner:
			raise ValueError("Frontend reads target the storage CPU; worker inputs use peer routing")
		if layer == 0:
			return select_rows(self.features, nodes)
		groups = self.catalog.selections(layer, nodes, self.catalog.version) if selections is None else selections
		if self.gather_inputs:
			return plane.gather_rows(groups, self.widths[layer], stack)
		if len(groups) == 1 and torch.equal(groups[0].positions, torch.arange(nodes.numel())):
			return plane.read_rows(groups[0].key, groups[0].rows, stack)
		output = torch.empty(nodes.numel(), self.widths[layer], dtype=torch.float32)
		for group in groups:
			values = plane.read_rows(group.key, group.rows, stack)
			output.index_copy_(0, group.positions, values)
		return output
