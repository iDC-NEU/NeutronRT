"""Retry derived computation after resetting a managed worker world."""

from contextlib import ExitStack, contextmanager
from time import perf_counter
from typing import Callable, Iterator, Optional, Protocol, Sequence, TypeVar

import torch
from dgl import DGLGraph
from torch import Tensor

from lib.graph.affected import AffectedLayer
from lib.graph.delta_plan import SparseLayerPlan
from lib.graph.ownership import Owner
from lib.graph.pipeline import PipelineTiming
from lib.graph.recompute import LayerPlan, RecomputePlan
from lib.graph.serving import UpdateResult, _PendingUpdate
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from lib.runtime.transport import ModelExecutionError, RemoteEndpoint, TransportFailure

Result = TypeVar("Result")


class WorkerWorld(Protocol):
	@property
	def reset_count(self) -> int: ...

	def healthy(self) -> bool: ...

	def reset(self) -> Sequence[RemoteEndpoint]: ...


class ResilientEmbeddingService(DistributedEmbeddingService):
	"""Retry model tasks and reads, while applying each root mutation only once.

	The inherited pipeline's coordinator owns the graph fence. Its compute thread
	can reset/retry without acquiring that fence from a second thread. Recovery
	never re-enters a public mutation or replays a caller's query consumer.
	"""

	def __init__(
		self,
		graph: DGLGraph,
		features: Tensor,
		widths: Sequence[int],
		plane: RemoteEmbeddingPlane,
		world: WorkerWorld,
		*,
		max_retries: int = 2,
		**kwargs,
	) -> None:
		if max_retries < 0:
			raise ValueError("Retry limit must be nonnegative")
		super().__init__(graph, features, widths, plane, **kwargs)
		self.world = world
		self.max_retries = max_retries
		self._recovery_depth = 0
		self._update_world: Optional[int] = None

	@contextmanager
	def _recovering(self):
		outer = self._recovery_depth == 0
		start, before = perf_counter(), self.counters.rows
		self._recovery_depth += 1
		try:
			yield
		finally:
			self._recovery_depth -= 1
			if outer:
				self.counters.recovery_seconds += perf_counter() - start
				self.counters.recovery_rows += self.counters.rows - before

	def _retry(self, action: Callable[[], Result], remaining: Optional[int] = None) -> Result:
		remaining = self.max_retries if remaining is None else remaining
		if self.world.healthy():
			try:
				return action()
			except (TransportFailure, ModelExecutionError):
				if remaining == 0:
					raise
		elif remaining == 0:
			raise TransportFailure("Managed worker world is unavailable and its retry limit is exhausted")
		with self._recovering():
			self.workers = list(self.world.reset())
			return self._retry(action, remaining - 1)

	@torch.inference_mode()
	def _run(self, nodes: Tensor, layer: int, prepared: Optional[RecomputePlan] = None) -> None:
		first = True

		def compute():
			nonlocal first
			plan = prepared if first else None
			first = False
			super(ResilientEmbeddingService, self)._run(nodes, layer, plan)

		self._retry(compute)

	def _run_wave(self, tasks: Sequence[LayerPlan]) -> None:
		self._retry(lambda: super(ResilientEmbeddingService, self)._run_wave(tasks))

	def _run_delta(self, layer: int, plan: SparseLayerPlan) -> None:
		self._retry(lambda: super(ResilientEmbeddingService, self)._run_delta(layer, plan))

	def _ensure_rows(self, nodes: Tensor, layer: int) -> None:
		with ExitStack() as recovery:
			for _ in range(self.max_retries + 1):
				before = self.world.reset_count
				self._retry(lambda: super(ResilientEmbeddingService, self)._ensure_rows(nodes, layer))
				if before == self.world.reset_count and self.world.healthy():
					return
				if self._recovery_depth == 0:
					recovery.enter_context(self._recovering())
			raise TransportFailure("Worker resets repeatedly interrupted requested-row computation")

	@contextmanager
	def _query_unfenced(self, nodes: Tensor, owner: Owner, layer: int, max_staleness: int = 0) -> Iterator[Tensor]:
		def load():
			stack = ExitStack()
			try:
				selections = self._prepare_query(nodes, layer, max_staleness)
				loader = super(ResilientEmbeddingService, self)._load
				values = (
					loader(layer, nodes, owner, stack)
					if selections is None
					else loader(layer, nodes, owner, stack, selections)
				)
				return values, stack
			except BaseException:
				stack.close()
				raise

		values, leases = self._retry(load)
		with leases:
			yield values

	def _finish_update(self, *args, **kwargs) -> UpdateResult:
		previous, self._update_world = self._update_world, self.world.reset_count
		try:
			return super()._finish_update(*args, **kwargs)
		finally:
			self._update_world = previous

	def _execute_merged(self, pending: _PendingUpdate) -> UpdateResult:
		previous, self._update_world = self._update_world, self.world.reset_count
		try:
			return super()._execute_merged(pending)
		finally:
			self._update_world = previous

	def _complete_update(self, pending: _PendingUpdate, timing: Optional[PipelineTiming] = None) -> UpdateResult:
		if self._update_world is None or self._update_world != self.world.reset_count or not self.world.healthy():
			# A later task's reset may discard already-published earlier chunks.
			# Revalidate all affected layers before crediting this update. Preserve
			# its applied-root record if recovery itself fails, so resume is safe.
			self._pending_update = pending
			with self._recovering():
				for _ in range(self.max_retries + 1):
					before = self.world.reset_count
					for affected in pending.layers:

						def ensure_layer(affected: AffectedLayer = affected) -> None:
							self._ensure_rows(affected.targets, affected.layer)

						self._retry(ensure_layer)
					if before == self.world.reset_count and self.world.healthy():
						break
				else:
					raise TransportFailure("Worker resets repeatedly interrupted update completion")
		return super()._complete_update(pending, timing)
