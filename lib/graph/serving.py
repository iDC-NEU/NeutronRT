"""Version-fenced embedding computation with GPU ownership and recomputation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Dict, Generic, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import torch
from dgl import DGLGraph
from torch import Tensor

from lib.graph.affected import AffectedLayer, AffectedPlanner
from lib.graph.catalog import EmbeddingCatalog, RowSelection
from lib.graph.data_plane import LocalEmbeddingPlane, LocalEndpoint
from lib.graph.neighborhood import NeighborhoodPolicy
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.graph.pipeline import PipelineTiming, execute_layer_pipeline
from lib.graph.policies import DispatchPolicy, FallbackCounters
from lib.graph.recompute import LayerPlan, RecomputePlan, RecomputePlanner, execute_recomputation

if TYPE_CHECKING:
	from lib.graph.deferred import DeferredUpdates

Apply = Callable[[LocalEndpoint, LayerPlan, Tensor], Tensor]


class ComputeEndpoint(Protocol):
	@property
	def owner(self) -> Owner: ...

	@property
	def device(self) -> torch.device: ...


Endpoint = TypeVar("Endpoint", bound=ComputeEndpoint)


@dataclass
class ExecutionCounters:
	tasks: int = 0
	rows: int = 0
	recovery_rows: int = 0
	recovery_seconds: float = 0.0
	updates: int = 0
	input_changes: int = 0
	update_seconds: float = 0.0
	staging_seconds: float = 0.0
	merged_batches: int = 0
	coalesced_affected_rows: int = 0
	wave_launches: int = 0
	wave_tasks_launched: int = 0
	wave_serial_fallbacks: int = 0
	reconstruction_tasks: int = 0
	reconstructed_rows: int = 0
	delta_tasks: int = 0
	delta_rows: int = 0
	delta_routed_edges: int = 0
	delta_fallback_tasks: int = 0
	delta_refresh_rows: int = 0
	delta_partial_refreshes: int = 0
	delta_workspace_fallbacks: int = 0
	delta_workspace_splits: int = 0


@dataclass(frozen=True)
class UpdateResult:
	version: int
	affected: Tuple[int, ...]
	tasks: int
	input_changes: int
	seconds: float
	preparation_seconds: float = 0.0
	compute_seconds: float = 0.0
	wait_seconds: float = 0.0
	lookahead: bool = False
	input_version: int = 0


@dataclass(frozen=True)
class AppliedMutation:
	"""An authoritative mutation applied once, before derived-state processing."""

	start: float
	changes: int
	changed_features: Optional[Tensor] = None
	edge_sources: Optional[Tensor] = None
	edge_targets: Optional[Tensor] = None


@dataclass(frozen=True)
class _PreparedLayer:
	affected: AffectedLayer
	first: Optional[LayerPlan]


@dataclass(frozen=True)
class _PendingUpdate:
	version: int
	changes: int
	start: float
	tasks_before: int
	layers: Tuple[AffectedLayer, ...]


def select_rows(tensor: Tensor, rows: Tensor) -> Tensor:
	"""Keep contiguous selections as views; gather only genuinely sparse rows."""
	if not rows.numel():
		return tensor.narrow(0, 0, 0)
	start = int(rows[0])
	if torch.equal(rows, torch.arange(start, start + rows.numel(), device=rows.device)):
		return tensor.narrow(0, start, rows.numel())
	return tensor.index_select(0, rows.to(tensor.device))


class EmbeddingService(Generic[Endpoint]):
	"""Storage-side coordinator; model callbacks execute on the chosen endpoint.

	The supplied CPU graph/features remain authoritative and are retained without
	cloning. Public query contexts fence mutation and retain data-plane leases.
	"""

	def __init__(
		self,
		graph: DGLGraph,
		features: Tensor,
		widths: Sequence[int],
		plane: LocalEmbeddingPlane,
		workers: Sequence[Endpoint],
		apply: Callable[[Endpoint, LayerPlan, Tensor], Tensor],
		*,
		batch_size: int = 8192,
		max_plan_edges: Optional[int] = None,
		source_degree_sensitive: bool = False,
		pipeline: bool = False,
		prefetch_edges: int = 1_000_000,
		preemption_enabled: bool = True,
		frontier_expander: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
		neighborhood_policy: Optional[NeighborhoodPolicy] = None,
		dispatch_policy: Optional[DispatchPolicy] = None,
		worker_feature_cache_bytes: int = 0,
	) -> None:
		if features.device.type != "cpu" or features.ndim != 2 or features.shape[0] != graph.num_nodes():
			raise ValueError("Storage requires one CPU feature row per graph node")
		if len(widths) < 2 or widths[0] != features.shape[1] or min(widths) < 1 or batch_size < 1:
			raise ValueError("Invalid layer widths or batch size")
		if not workers or len({node.owner for node in workers}) != len(workers):
			raise ValueError("Need distinct live compute endpoints")
		if prefetch_edges < 0:
			raise ValueError("Prefetch edge budget must be nonnegative")
		if type(worker_feature_cache_bytes) is not int or worker_feature_cache_bytes < 0:
			raise ValueError("Worker feature cache budget must be a nonnegative integer")
		if worker_feature_cache_bytes:
			raise NotImplementedError(
				"Worker feature-row reuse is fail-closed until miss-only transport is implemented"
			)
		self.graph = graph
		self.features = features
		self.widths = tuple(widths)
		self.plane = plane
		self.workers = list(workers)
		self.apply = apply
		self.batch_size = batch_size
		self.source_degree_sensitive = source_degree_sensitive
		self.pipeline = pipeline
		self.prefetch_edges = prefetch_edges
		self.preemption_enabled = preemption_enabled
		self.checkpoint: Optional[Callable[[], None]] = None
		self._fence = RLock()
		self._next = 0
		self.catalog = EmbeddingCatalog(graph.num_nodes(), len(widths) - 1, plane.directory)
		self._additional_catalogs: Tuple[EmbeddingCatalog, ...] = ()
		self.planner = RecomputePlanner(
			graph,
			len(widths) - 1,
			max_edges=max_plan_edges,
			batch_size=batch_size,
			source_degree_sensitive=source_degree_sensitive,
			policy=neighborhood_policy,
		)
		self.affected = AffectedPlanner(
			graph, len(widths) - 1, source_degree_sensitive=source_degree_sensitive, expander=frontier_expander
		)
		self.counters = ExecutionCounters()
		self._pending_update: Optional[_PendingUpdate] = None
		self._last_update: Optional[UpdateResult] = None
		self.deferred: Optional[DeferredUpdates] = None
		self.dispatch_policy = dispatch_policy or DispatchPolicy()
		self.worker_feature_cache_bytes = worker_feature_cache_bytes
		self.fallbacks = FallbackCounters()
		self._dispatch_reservations: Dict[Owner, List[int]] = {}

	def _state_catalogs(self) -> Tuple[EmbeddingCatalog, ...]:
		return (self.catalog, *self._additional_catalogs)

	def _state_available(self, layer: int, nodes: Tensor, version: int) -> Tensor:
		valid = self.catalog.available(layer, nodes, version)
		for catalog in self._additional_catalogs:
			valid &= catalog.available(layer, nodes, version)
		return valid

	def _pending_analysis(self) -> Tuple[int, ...]:
		return tuple(sorted({layer for catalog in self._state_catalogs() for layer in catalog.pending_layers}))

	def _begin_version(self, version: int, dirty: Dict[int, Tensor], *, pending_layers: Sequence[int] = ()) -> None:
		catalogs = self._state_catalogs()
		if any(catalog.version != self.catalog.version or catalog.pending_layers for catalog in catalogs):
			raise RuntimeError("Derived catalog versions disagree or analysis is incomplete")
		for catalog in catalogs:
			catalog.begin_version(version, dirty, pending_layers=pending_layers)

	def _finish_analysis(self, layer: int, dirty: Tensor, version: int) -> None:
		pending = [catalog for catalog in self._state_catalogs() if layer in catalog.pending_layers]
		if not pending:
			raise ValueError("Layer analysis is not pending")
		for catalog in pending:
			catalog.finish_analysis(layer, dirty, version)

	def _locality_rows(self, nodes: Tensor, input_layer: int) -> Dict[Owner, int]:
		if input_layer <= 0 or not nodes.numel():
			return {}
		available = self.catalog.available(input_layer, nodes, self.catalog.version)
		if not available.any():
			return {}
		nodes = nodes[available]
		rows: Dict[Owner, int] = {}
		for group in self.catalog.selections(input_layer, nodes, self.catalog.version):
			owner = self.plane.directory.locate(group.key).owner
			rows[owner] = rows.get(owner, 0) + group.rows.numel()
		return rows

	def _pick(
		self,
		*,
		required: Optional[Tensor] = None,
		input_layer: int = 0,
		task_cost: Optional[int] = None,
		candidates: Optional[Sequence[Endpoint]] = None,
	) -> Endpoint:
		workers = self.workers if candidates is None else list(candidates)
		if not workers:
			raise RuntimeError("No live compute endpoints")
		required = required if required is not None else torch.empty(0, dtype=torch.int64)
		cost = max(1, required.numel()) if task_cost is None else task_cost
		locality_rows = None
		if self.dispatch_policy.mode != "round_robin":
			locality_rows = self._locality_rows(required, input_layer)
		choice = self.dispatch_policy.choose(
			workers,
			locality_rows=locality_rows,
			total_rows=required.numel(),
			task_cost=cost,
		)
		self._dispatch_reservations.setdefault(choice.worker.owner, []).append(choice.reserved_cost)
		return choice.worker

	def _adjust_dispatch(self, worker: Endpoint, actual_cost: int) -> None:
		reservations = self._dispatch_reservations.get(worker.owner)
		if not reservations:
			raise RuntimeError("Dispatch task was not reserved")
		reserved = reservations[-1]
		self.dispatch_policy.adjust(worker.owner, reserved, actual_cost)
		reservations[-1] = actual_cost

	def _release_dispatch(self, worker: Endpoint) -> None:
		reservations = self._dispatch_reservations.get(worker.owner)
		if not reservations:
			raise RuntimeError("Dispatch task was not reserved")
		reserved = reservations.pop()
		self.dispatch_policy.release(worker.owner, reserved)
		if not reservations:
			self._dispatch_reservations.pop(worker.owner, None)

	@staticmethod
	def _task_cost(task: LayerPlan) -> int:
		return int(max(1, task.targets.numel() + task.block.num_edges()))

	def profile(self) -> dict:
		return {
			"counters": {name: value for name, value in vars(self.counters).items() if type(value) is int},
			"dispatch": self.dispatch_policy.profile(),
			"worker_feature_cache": {
				"enabled": False,
				"budget_bytes": self.worker_feature_cache_bytes,
				"status": "disabled" if not self.worker_feature_cache_bytes else "unsupported_miss_only_protocol",
			},
			"fallback_reasons": self.fallbacks.as_dict(),
			"deferred": self.deferred.profile() if self.deferred is not None else None,
		}

	def _load(
		self,
		layer: int,
		nodes: Tensor,
		owner: Owner,
		stack: ExitStack,
		selections: Optional[Tuple[RowSelection, ...]] = None,
	) -> Tensor:
		if layer == 0:
			return self.plane.to_owner(select_rows(self.features, nodes), owner)
		groups = self.catalog.selections(layer, nodes, self.catalog.version) if selections is None else selections
		if len(groups) == 1 and torch.equal(groups[0].positions, torch.arange(nodes.numel())):
			values = stack.enter_context(self.plane.acquire(groups[0].key, owner=owner))
			return select_rows(values, groups[0].rows)
		output = self.plane.empty(owner, (nodes.numel(), self.widths[layer]), dtype=self.features.dtype)
		for group in groups:
			values = stack.enter_context(self.plane.acquire(group.key, owner=owner))
			selected = select_rows(values, group.rows)
			output.index_copy_(0, group.positions.to(output.device), selected)
		return output

	@torch.inference_mode()
	def _run(self, nodes: Tensor, layer: int, prepared: Optional[RecomputePlan] = None) -> None:
		version = self.catalog.version
		plans = (
			(prepared,)
			if prepared is not None
			else self.planner.plans(nodes, layer, version, available=self._state_available)
		)
		for plan in plans:
			try:
				self._execute_plan(plan)
			finally:
				if not self.planner.policy.reuse_intermediates:
					for catalog in self._state_catalogs():
						for intermediate in range(1, plan.layer):
							for key in catalog.discard_layer(intermediate, plan.version):
								self.plane.retire(key)

	def _execute_plan(self, plan: RecomputePlan) -> None:
		version = plan.version
		pending: List[StateKey] = []
		attempts: Dict[StateKey, ComputeAttempt] = {}
		worker = self.workers[0] if self.workers else None
		reserved_cost: Optional[int] = None
		with ExitStack() as stack:

			def load(input_layer: int, required: Tensor, requested: int) -> Tensor:
				nonlocal worker, reserved_cost
				if requested != self.catalog.version:
					raise ValueError("Graph version changed during execution")
				worker = self._pick(required=required, input_layer=input_layer, task_cost=max(1, required.numel()))
				reserved_cost = max(1, required.numel())
				return self._load(input_layer, required, worker.owner, stack)

			def apply(task: LayerPlan, inputs: Tensor) -> Tensor:
				nonlocal reserved_cost
				if worker is None:
					raise RuntimeError("No compute endpoint selected")
				actual_cost = self._task_cost(task)
				self._adjust_dispatch(worker, actual_cost)
				reserved_cost = actual_cost
				key = self.catalog.reserve(task.layer, task.targets, version)
				pending.append(key)
				attempts[key] = self.plane.directory.begin(key, worker.owner)
				return self.apply(worker, task, inputs)

			def publish(output_layer: int, output_nodes: Tensor, values: Tensor, requested: int) -> None:
				nonlocal reserved_cost
				if worker is None or requested != self.catalog.version:
					raise ValueError("Publication graph version changed during execution")
				if values.shape[1] != self.widths[output_layer] or values.dtype != self.features.dtype:
					raise ValueError("Computed embedding width or dtype does not match the model")
				key = pending[-1]
				attempt = attempts[key]
				self.plane.publish(attempt, values)
				obsolete = self.catalog.publish(key)
				pending.pop()
				del attempts[key]
				self.counters.tasks += 1
				self.counters.rows += output_nodes.numel()
				self._release_dispatch(worker)
				reserved_cost = None
				for retired in obsolete:
					self.plane.retire(retired)

			try:
				execute_recomputation(plan, load=load, apply=apply, publish=publish)
			finally:
				for key in pending:
					self.catalog.cancel(key)
					self.plane.retire(key)
				if worker is not None and reserved_cost is not None:
					self._release_dispatch(worker)

	def _ensure_rows(self, nodes: Tensor, layer: int) -> None:
		"""Caller holds the graph fence, including on the pipeline coordinator thread."""
		self.catalog.available(layer, nodes, self.catalog.version)
		for chunk in nodes.unique(sorted=True).split(self.batch_size):
			if not self._state_available(layer, chunk, self.catalog.version).all():
				self._run(chunk, layer)
				self._yield_compute()

	def _yield_compute(self) -> None:
		if self.checkpoint is not None:
			self.checkpoint()

	def ensure(self, nodes: Tensor, layer: Optional[int] = None) -> None:
		with self._fence:
			self._ensure_rows(nodes, len(self.widths) - 1 if layer is None else layer)

	def _prepare_layer(self, affected: AffectedLayer) -> _PreparedLayer:
		# Keep only one prefetched block per layer. Large frontiers remain ID
		# metadata; allocating every block ahead would exhaust storage memory.
		first = affected.targets[: self.batch_size]
		block = None
		limit = self.prefetch_edges
		response_bytes = getattr(self.graph, "response_bytes", None)
		if response_bytes is not None:
			limit = min(limit, response_bytes // 8 - first.numel())
		if first.numel() and int(self.graph.in_degrees(first).sum()) <= limit:
			# The bounded dependency planner will partition a rejected response.
			with suppress(MemoryError):
				block = self.affected.block(affected.layer, first, max_edges=self.planner.max_edges)
		return _PreparedLayer(affected, block)

	def _compute_layer(self, prepared: _PreparedLayer) -> None:
		layer, targets = prepared.affected.layer, prepared.affected.targets
		for index, chunk in enumerate(targets.split(self.batch_size)):
			if self._state_available(layer, chunk, self.catalog.version).all():
				continue
			task = prepared.first if index == 0 else None
			if task is None:
				self._run(chunk, layer)
				self._yield_compute()
				continue
			if layer > 1:
				self._ensure_rows(task.sources, layer - 1)
			plan = RecomputePlan(self.catalog.version, layer, chunk, task.targets, (task,))
			self._run(chunk, layer, plan)
			self._yield_compute()

	@contextmanager
	def _query_unfenced(self, nodes: Tensor, owner: Owner, layer: int, max_staleness: int = 0) -> Iterator[Tensor]:
		# Used only while the coordinator holds the graph fence, including when
		# its compute thread services a high-priority request at a chunk boundary.
		with ExitStack() as stack:
			selections = self._prepare_query(nodes, layer, max_staleness)
			yield (
				self._load(layer, nodes, owner, stack)
				if selections is None
				else self._load(layer, nodes, owner, stack, selections)
			)

	def _check_query_budget(self, max_staleness: int) -> None:
		if type(max_staleness) is not int or max_staleness < 0:
			raise ValueError("Query staleness must be a nonnegative integer")
		if max_staleness and (self.deferred is None or max_staleness > self.deferred.max_versions):
			raise ValueError("Query staleness exceeds the enabled shared version budget")

	def _prepare_query(self, nodes: Tensor, layer: int, max_staleness: int) -> Optional[Tuple[RowSelection, ...]]:
		self._check_query_budget(max_staleness)
		if not max_staleness or layer == 0:
			self._ensure_rows(nodes, layer)
			return None
		servable = self.catalog.servable(layer, nodes, self.catalog.version, max_staleness=max_staleness)
		self._ensure_rows(nodes[~servable], layer)
		return self.catalog.relaxed_selections(layer, nodes, self.catalog.version, max_staleness=max_staleness)

	@contextmanager
	def query(
		self, nodes: Tensor, owner: Owner, *, layer: Optional[int] = None, max_staleness: int = 0
	) -> Iterator[Tensor]:
		with self._fence:
			layer_ = len(self.widths) - 1 if layer is None else layer
			with self._query_unfenced(nodes, owner, layer_, max_staleness) as values:
				yield values

	def recover(self, owner: Owner) -> Tuple[StateKey, ...]:
		with self._fence:
			if owner not in {node.owner for node in self.workers}:
				raise ValueError("Recovery currently supports compute endpoint loss")
			start = perf_counter()
			lost = self.plane.lose_owner(owner)
			self.workers = [node for node in self.workers if node.owner != owner]
			before = self.counters.rows
			for key in sorted(lost, key=lambda key: key.layer):
				nodes = self.catalog.current_nodes(key)
				if nodes.numel():
					self.ensure(nodes, key.layer)
			self.counters.recovery_rows += self.counters.rows - before
			self.counters.recovery_seconds += perf_counter() - start
			return lost

	def _finish_update(
		self,
		start: float,
		changes: int,
		*,
		changed_features: Optional[Tensor] = None,
		edge_sources: Optional[Tensor] = None,
		edge_targets: Optional[Tensor] = None,
	) -> UpdateResult:
		version = self.catalog.version + 1
		self._begin_version(version, {}, pending_layers=range(1, len(self.widths)))
		analyzed = set()
		before = self.counters.tasks
		layers: List[AffectedLayer] = []

		def work() -> Iterator[AffectedLayer]:
			for affected in self.affected.analyze(
				changed_features=changed_features,
				edge_targets=edge_targets,
				degree_sources=edge_sources if self.source_degree_sensitive else None,
			):
				self._finish_analysis(affected.layer, affected.targets, version)
				analyzed.add(affected.layer)
				layers.append(affected)
				yield affected

		try:
			# Materialize DGL's lazy adjacency formats before concurrent read-only
			# analysis. The authoritative graph cannot mutate until both threads drain.
			timing = self._execute_layers(work())
		except BaseException:
			all_nodes = torch.arange(self.graph.num_nodes())
			known = {item.layer: item for item in layers}
			fallback = tuple(known.get(layer, AffectedLayer(layer, all_nodes)) for layer in range(1, len(self.widths)))
			self._pending_update = _PendingUpdate(version, changes, start, before, fallback)
			raise
		finally:
			# If planning fails after the root mutation, fresh queries must never
			# reuse rows whose validity was not established for the new graph.
			for layer in range(1, len(self.widths)):
				if layer not in analyzed:
					self._finish_analysis(layer, torch.arange(self.graph.num_nodes()), version)
		pending = _PendingUpdate(version, changes, start, before, tuple(layers))
		return self._complete_update(pending, timing)

	def _execute_layers(self, layers: Iterator[AffectedLayer]) -> PipelineTiming:
		self.graph.create_formats_()
		if not self.planner.policy.reuse_intermediates:
			# Resolve validity for all layers first, then expand backward from the
			# final affected outputs. No old intermediate embedding truncates the DAG.
			started = perf_counter()
			affected = tuple(layers)
			timing = PipelineTiming(prepare_seconds=perf_counter() - started)
			if affected:
				started = perf_counter()
				for chunk in affected[-1].targets.split(self.batch_size):
					for plan in self.planner.plans(chunk, affected[-1].layer, self.catalog.version):
						self._execute_recompute_plan(plan)
					self._yield_compute()
				timing.compute_seconds = perf_counter() - started
				timing.layers = len(affected)
			return timing
		with ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-compute") as executor:
			return execute_layer_pipeline(
				layers, self._prepare_layer, self._compute_layer, executor=executor, lookahead=self.pipeline
			)

	def _execute_recompute_plan(self, plan: RecomputePlan) -> None:
		self._run(plan.requested, plan.layer, plan)

	def _execute_merged(self, pending: _PendingUpdate) -> UpdateResult:
		timing = self._execute_layers(iter(pending.layers))
		return self._complete_update(pending, timing)

	def _complete_update(self, pending: _PendingUpdate, timing: Optional[PipelineTiming] = None) -> UpdateResult:
		elapsed = perf_counter() - pending.start
		result = UpdateResult(
			pending.version,
			tuple(item.targets.numel() for item in pending.layers),
			self.counters.tasks - pending.tasks_before,
			pending.changes,
			elapsed,
			preparation_seconds=timing.prepare_seconds if timing else 0.0,
			compute_seconds=timing.compute_seconds if timing else 0.0,
			wait_seconds=timing.wait_seconds if timing else 0.0,
			lookahead=self.pipeline if timing else False,
			input_version=pending.version,
		)
		self.counters.updates += 1
		self.counters.input_changes += pending.changes
		self.counters.update_seconds += elapsed
		self._pending_update = None
		self._last_update = result
		if self.deferred is not None:
			self.deferred.completed(pending, result)
		return result

	def _require_update_slot(self, *, allow_deferred: bool = False) -> None:
		if self._pending_update is not None or self._pending_analysis():
			raise RuntimeError("Previous update is incomplete; recover and resume it before applying another mutation")
		if not allow_deferred and self.deferred is not None:
			self.deferred.flush()

	def resume_update(self, version: Optional[int] = None) -> UpdateResult:
		"""Finish an applied mutation without applying its input changes again."""
		with self._fence:
			pending = self._pending_update
			if pending is None:
				if self._last_update is not None and (version is None or version == self._last_update.version):
					return self._last_update
				raise ValueError("No matching update to resume")
			if pending.version != self.catalog.version or version is not None and version != pending.version:
				raise ValueError("Resume graph version does not match the applied update")
			if self._pending_analysis():
				all_nodes = torch.arange(self.graph.num_nodes())
				for layer in self._pending_analysis():
					self._finish_analysis(layer, all_nodes, pending.version)
			for affected in pending.layers:
				self.ensure(affected.targets, affected.layer)
			return self._complete_update(pending)

	def change_features(self, nodes: Tensor, values: Tensor) -> UpdateResult:
		with self._fence:
			self._require_update_slot()
			mutation = self._mutate_features(nodes, values)
			return self._finish_mutation(mutation)

	def _mutate_features(self, nodes: Tensor, values: Tensor) -> AppliedMutation:
		"""Caller holds the graph fence and an available update slot."""
		start = perf_counter()
		self.catalog.available(0, nodes, self.catalog.version)
		if nodes.unique().numel() != nodes.numel():
			raise ValueError("Feature update node IDs must be unique")
		if values.device.type != "cpu" or values.dtype != self.features.dtype:
			raise ValueError("Feature updates must match the storage CPU dtype")
		if values.shape != (nodes.numel(), self.features.shape[1]):
			raise ValueError("Feature update shape does not match the requested nodes")
		base, source = self.features.untyped_storage(), values.untyped_storage()
		base_start, source_start = base.data_ptr(), source.data_ptr()
		if base_start < source_start + source.nbytes() and source_start < base_start + base.nbytes():
			values = values.clone()
		self.features.index_copy_(0, nodes, values)
		return AppliedMutation(start, nodes.numel(), changed_features=nodes)

	def _finish_mutation(self, mutation: AppliedMutation) -> UpdateResult:
		return self._finish_update(
			mutation.start,
			mutation.changes,
			changed_features=mutation.changed_features,
			edge_sources=mutation.edge_sources,
			edge_targets=mutation.edge_targets,
		)

	def _edge_pairs(self, sources: Tensor, targets: Tensor) -> None:
		self.catalog.available(0, sources, self.catalog.version)
		self.catalog.available(0, targets, self.catalog.version)
		if sources.shape != targets.shape:
			raise ValueError("Edge endpoints must have equal lengths")

	def insert_edges(self, sources: Tensor, targets: Tensor, *, attributes: Optional[Tensor] = None) -> UpdateResult:
		with self._fence:
			self._require_update_slot()
			return self._finish_mutation(self._mutate_edges(sources, targets, delete=False, attributes=attributes))

	def delete_edges(self, sources: Tensor, targets: Tensor) -> UpdateResult:
		"""Remove all matching parallel edges; absent pairs are an idempotent no-op."""
		with self._fence:
			self._require_update_slot()
			return self._finish_mutation(self._mutate_edges(sources, targets, delete=True))

	def _mutate_edges(
		self, sources: Tensor, targets: Tensor, *, delete: bool, attributes: Optional[Tensor] = None
	) -> AppliedMutation:
		start = perf_counter()
		self._edge_pairs(sources, targets)
		if not delete:
			existing = getattr(self.graph, "edata", {}).get("nrt")
			if attributes is not None:
				if (
					attributes.device.type != "cpu"
					or attributes.dtype != torch.float32
					or attributes.ndim != 2
					or attributes.shape[0] != sources.numel()
					or not torch.isfinite(attributes).all()
				):
					raise ValueError("Edge attributes must be a finite CPU float32 matrix with one row per edge")
				if existing is None or attributes.shape[1] != existing.shape[1]:
					raise ValueError("Initialize the graph's nrt edge schema before inserting attributed edges")
			elif existing is not None:
				attributes = existing.new_zeros((sources.numel(), existing.shape[1]))
				if attributes.shape[1] > 1:
					attributes[:, 1] = 1
			if attributes is None:
				self.graph.add_edges(sources, targets)
			else:
				self.graph.add_edges(sources, targets, data={"nrt": attributes})
			return AppliedMutation(start, sources.numel(), edge_sources=sources, edge_targets=targets)
		exists = self.graph.has_edges_between(sources, targets)
		removed_sources, removed_targets = sources[exists], targets[exists]
		if exists.any():
			pair_delete = getattr(self.graph, "delete_pairs", None)
			if pair_delete is not None:
				pair_delete(removed_sources, removed_targets)
			else:
				_, _, identifiers = self.graph.edge_ids(removed_sources, removed_targets, return_uv=True)
				self.graph.remove_edges(identifiers.unique())
		return AppliedMutation(start, sources.numel(), edge_sources=removed_sources, edge_targets=removed_targets)
