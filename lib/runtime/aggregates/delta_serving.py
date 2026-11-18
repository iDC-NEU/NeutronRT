"""Storage-planned sparse updates, with selected full-neighbor fallbacks."""

from contextlib import ExitStack
from typing import TYPE_CHECKING, Sequence, Union

import torch
from torch import Tensor

from lib.graph.delta_plan import SparseLayerPlan, compile_delta_routes, select_delta_routes, slice_delta_routes
from lib.graph.ownership import MissingState, Owner, StateKey
from lib.graph.recompute import PlanTooLarge
from lib.runtime.aggregates.delta import delta_workspace, pack_delta_compute
from lib.runtime.aggregates.delta_inputs import DeltaInputs
from lib.runtime.aggregates.delta_recipes import AggregateDeltaRecipes
from lib.runtime.aggregates.delta_state import DeltaState
from lib.runtime.aggregates.tables import InputTable, supply_tables

if TYPE_CHECKING:
	from lib.graph.serving import _PreparedLayer


class DeltaExecution:
	def __init__(self, state: DeltaState) -> None:
		self.state, self.service = state, state.service
		self.inputs = DeltaInputs(self.service)

	def full(self, layer: int, nodes: Tensor, *, refresh: bool = False, reason: str) -> None:
		service = self.service
		service._run(nodes, layer)
		service.counters.delta_fallback_tasks += 1
		service.fallbacks.add(layer, reason, nodes.numel())
		if refresh:
			service.counters.delta_refresh_rows += nodes.numel()

	def compute(self, prepared: "_PreparedLayer") -> None:
		service, context = self.service, self.state.active
		assert context is not None
		layer, targets = prepared.affected.layer, prepared.affected.targets
		plan = None
		fallback_reason = "cold_or_unavailable_state"
		if context.enabled:
			changed = (
				context.mutation.features.nodes
				if layer == 1 and context.mutation.features is not None
				else torch.empty(0, dtype=torch.int64)
				if layer == 1
				else context.layers[layer - 1].embeddings.snapshot.nodes
			)
			edges = context.mutation.edges
			try:
				plan = compile_delta_routes(
					service.graph,
					changed,
					edges.sources,
					edges.targets,
					edges.weights,
					max_edges=service.planner.max_edges if service.planner.max_edges is not None else 1_000_000,
				)
			except PlanTooLarge:
				fallback_reason = "plan_edge_budget"
			if plan is not None and not torch.equal(plan.targets, targets):
				plan = None
				fallback_reason = "incomplete_delta_target_coverage"
		for start in range(0, targets.numel(), service.batch_size):
			stop = min(start + service.batch_size, targets.numel())
			nodes = targets[start:stop]
			if service._state_available(layer, nodes, service.catalog.version).all():
				continue
			if plan is None:
				self.full(layer, nodes, reason=fallback_reason)
			else:
				service._run_delta(layer, slice_delta_routes(plan, start, stop))
			service._yield_compute()

	@torch.inference_mode()
	def run(self, layer: int, plan: SparseLayerPlan) -> None:
		service, context = self.service, self.state.active
		if context is None or not context.enabled or layer not in context.layers:
			return self.full(layer, plan.targets, reason="cold_or_unavailable_state")
		anchors = context.layers[layer].aggregates
		selected = anchors.lookup(plan.targets)
		if not selected.available.all():
			return self.full(layer, plan.targets, reason="missing_aggregate_anchor")
		next_depth: Union[int, Tensor]
		if self.state.row_refresh:
			depth = self.state.row_depths(selected.selections, plan.targets.numel())
			expired = depth >= self.state.max_steps
			if expired.any():
				self.full(layer, plan.targets[expired], refresh=True, reason="periodic_refresh")
				if not expired.all():
					service.counters.delta_partial_refreshes += 1
					self.run(layer, select_delta_routes(plan, (~expired).nonzero().flatten()))
				return
			next_depth = depth + 1
		else:
			maximum = self.state.depth([group.key for group in selected.selections])
			if maximum >= self.state.max_steps:
				return self.full(layer, plan.targets, refresh=True, reason="periodic_refresh")
			next_depth = maximum + 1
		recipes = None
		if layer > 1 and service.aggregate_only:
			recipes = AggregateDeltaRecipes(service, context, layer, plan)
			try:
				recipes.prepare()
			except MissingState:
				return self.full(layer, plan.targets, reason="missing_aggregate_anchor")
		elif layer > 1:
			previous = context.layers[layer - 1].embeddings
			if not previous.lookup(plan.inputs.value_nodes).available.all():
				return self.full(layer, plan.targets, reason="missing_previous_embedding")
			try:
				self.inputs.prepare_old(previous, plan.edges.value_nodes)
			except MissingState:
				return self.full(layer, plan.targets, reason="missing_previous_embedding")
			service._ensure_rows(torch.cat((plan.targets, plan.inputs.value_nodes)).unique(sorted=True), layer - 1)
		plane = service.remote_plane
		task_cost = max(1, plan.targets.numel() + plan.inputs.sources.numel() + plan.edges.sources.numel())
		worker = service._pick(
			required=plan.inputs.value_nodes,
			input_layer=layer - 1,
			task_cost=task_cost,
		)
		try:
			output_bytes = 2 * plan.targets.numel() * service.widths[layer] * 4
			required = output_bytes
			if recipes is not None:
				required += recipes.workspace_bytes
				if required > worker.max_bytes:
					service.counters.delta_workspace_fallbacks += 1
					return self.full(layer, plan.targets, reason="workspace_capacity")
			with ExitStack() as temporary:
				tables: Sequence[InputTable]
				if recipes is not None:
					workspace = temporary.enter_context(
						recipes.temporary.workspace(worker.owner, required, output_bytes=output_bytes)
					)
					tables = recipes.tables(worker.owner, temporary, workspace)
				else:
					tables = self._retained_tables(layer, plan, worker.owner)
					admission = delta_workspace(plan, service.widths, layer, tables)
					# CPU row budgets bound retained cache, not host tensor working memory.
					# CUDA workers must leave device headroom before receiving any inputs.
					cuda = worker.device.type == "cuda"
					if cuda and admission.required_bytes > worker.max_bytes or admission.frame_bytes > 512 * 1024**2:
						if plan.targets.numel() < 2:
							service.counters.delta_workspace_fallbacks += 1
							return self.full(layer, plan.targets, reason="workspace_capacity")
						service.counters.delta_workspace_splits += 1
						middle = plan.targets.numel() // 2
						# Release CPU feature selections before recursively sizing child plans.
						del tables
						self.run(layer, slice_delta_routes(plan, 0, middle))
						self.run(layer, slice_delta_routes(plan, middle, plan.targets.numel()))
						return
					plane.make_room(worker.owner, admission.required_bytes if cuda else output_bytes)
					# Admission can move retained rows to CPU; refresh their certified routes.
					tables = self._retained_tables(layer, plan, worker.owner)
				self._submit(layer, plan, worker.owner, tables, next_depth)
		finally:
			service._release_dispatch(worker)

	def _retained_tables(self, layer: int, plan: SparseLayerPlan, target: Owner) -> Sequence[InputTable]:
		context = self.state.active
		assert context is not None
		previous = context.layers[layer - 1].embeddings if layer > 1 else None
		# Resolve locations after making room: retained old keys can move to CPU.
		return (
			self.inputs.old_sum(context.layers[layer].aggregates, plan.targets, target),
			self.inputs.current(layer - 1, plan.targets, target),
			self.inputs.current(layer - 1, plan.inputs.value_nodes, target),
			self.inputs.features_old(plan.inputs.value_nodes, context.mutation.features)
			if previous is None
			else self.inputs.old(previous, plan.inputs.value_nodes, target),
			self.inputs.features_old(plan.edges.value_nodes, context.mutation.features)
			if previous is None
			else self.inputs.old(previous, plan.edges.value_nodes, target),
		)

	def _submit(
		self,
		layer: int,
		plan: SparseLayerPlan,
		target: Owner,
		tables: Sequence[InputTable],
		next_depth: Union[int, Tensor],
	) -> None:
		service, plane = self.service, self.service.remote_plane
		degrees = service.graph.in_degrees(plan.targets).clone()
		assert plan.edges.weights is not None
		degrees.index_add_(0, plan.edges.targets, -plan.edges.weights)
		if (degrees < 0).any():
			raise RuntimeError("Structural corrections disagree with the prior aggregate degree")
		catalogs = service._state_catalogs()
		keys: list[StateKey] = []
		published: set[StateKey] = set()
		obsolete: list[StateKey] = []
		try:
			for catalog in catalogs:
				keys.append(catalog.reserve(layer, plan.targets, service.catalog.version))
			h, a = (plane.directory.begin(key, target) for key in keys)
			packet = pack_delta_compute(plan, h, a, service.widths, tables, degrees)
			with ExitStack() as leases:
				results = plane.transport.compute_delta(
					target,
					packet,
					output_width=service.widths[layer],
					supply_inputs=lambda: supply_tables(plane, tables, target, leases, gather=service.gather_inputs),
				)
				for attempt, result in zip((h, a), results, strict=True):
					plane.publish_result(attempt, result)
				for catalog, key in zip(catalogs, keys, strict=True):
					obsolete.extend(catalog.publish(key))
					published.add(key)
			self.state.record(a.key, next_depth, obsolete)
			service.counters.tasks += 1
			service.counters.rows += plan.targets.numel()
			service.counters.delta_tasks += 1
			service.counters.delta_rows += plan.targets.numel()
			service.counters.delta_routed_edges += plan.inputs.sources.numel() + plan.edges.sources.numel()
		finally:
			for catalog, key in zip(catalogs, keys, strict=False):
				if key not in published:
					catalog.cancel(key)
					plane.retire(key)
			for key in obsolete:
				plane.retire(key)
