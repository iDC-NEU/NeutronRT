"""Bounded waves: prepare inputs while peers are idle, then run models together."""

from contextlib import ExitStack, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from lib.graph.ownership import ComputeAttempt, Location, StateKey
from lib.graph.recompute import LayerPlan
from lib.graph.serving import select_rows
from lib.runtime.protocol import InputChunk, pack_compute
from lib.runtime.transport import PendingCompute, PreparedCompute, ReadyResult, RemoteEndpoint, TransportFailure

if TYPE_CHECKING:
	from lib.runtime.service import DistributedEmbeddingService


@dataclass
class _Output:
	task: LayerPlan
	worker: RemoteEndpoint
	key: StateKey
	attempt: ComputeAttempt
	prepared: Optional[PreparedCompute] = None
	pending: Optional[PendingCompute] = None
	result: Optional[ReadyResult] = None
	registered: bool = False
	published: bool = False


def run_wave(service: "DistributedEmbeddingService", tasks: Sequence[LayerPlan]) -> None:
	"""One task per worker, no model/peer-input cycles, and no unclaimed outputs."""
	if not tasks:
		return
	if len(tasks) > len(service.workers) or len({task.layer for task in tasks}) != 1:
		raise ValueError("A wave needs at most one same-layer task per worker")
	for task in tasks:
		if task.layer > 1:
			service._ensure_rows(task.sources, task.layer - 1)
	# A reset while ensuring a later input can invalidate an earlier dependency.
	# Retry the wave through the normal managed-world recovery path.
	if any(
		task.layer > 1 and not service.catalog.available(task.layer - 1, task.sources, service.catalog.version).all()
		for task in tasks
	):
		raise TransportFailure("Worker reset invalidated a prepared wave dependency")
	tasks = [
		task
		for task in tasks
		if not service.planner.policy.reuse_intermediates
		or not service.catalog.available(task.layer, task.targets, service.catalog.version).all()
	]
	if not tasks:
		return
	if len(tasks) == 1:
		task = tasks[0]
		service.counters.wave_serial_fallbacks += 1
		for block in service._split_task(task):
			service._run_task(block, service.catalog.version)
		return
	plane, transport = service.remote_plane, service.remote_plane.transport
	workers = []
	remaining = list(service.workers)
	try:
		for task in tasks:
			worker = service._pick(
				required=task.sources,
				input_layer=task.layer - 1,
				task_cost=service._task_cost(task),
				candidates=remaining,
			)
			workers.append(worker)
			remaining.remove(worker)
	except BaseException:
		for worker in workers:
			service._release_dispatch(worker)
		raise
	if len({worker.owner for worker in workers}) != len(workers):
		for worker in workers:
			service._release_dispatch(worker)
		raise ValueError("A wave cannot assign two tasks to the same worker")
	# Writeback/eviction must finish before prepared tasks pin their input chunks.
	try:
		for task, worker in zip(tasks, workers, strict=True):
			plane.make_room(worker.owner, task.targets.numel() * service.widths[task.layer] * 4)
	except BaseException:
		for worker in workers:
			service._release_dispatch(worker)
		raise
	released = False

	def release_workers() -> None:
		nonlocal released
		if released:
			return
		released = True
		for worker in workers:
			service._release_dispatch(worker)

	outputs = []
	failure: Optional[BaseException] = None
	with ExitStack() as stack:
		try:
			for task, worker in zip(tasks, workers, strict=True):
				key = service.catalog.reserve(task.layer, task.targets, service.catalog.version)
				try:
					attempt = plane.directory.begin(key, worker.owner)
				except BaseException:
					service.catalog.cancel(key)
					plane.retire(key)
					raise
				output = _Output(task, worker, key, attempt)
				outputs.append(output)
				inputs = []
				if task.layer > 1:
					for group in service.catalog.selections(task.layer - 1, task.sources, service.catalog.version):
						location = plane.directory.locate(group.key, preferred=worker.owner)
						proxy = not service.peer_fetch and location.owner not in (worker.owner, plane.storage.owner)
						inputs.append(InputChunk(location, group.rows, group.positions, sender=0 if proxy else None))
				packet = pack_compute(
					task,
					attempt,
					width=service.widths[task.layer - 1],
					inputs=inputs,
					features=select_rows(service.features, task.sources) if task.layer == 1 else None,
				)

				def supply(inputs=inputs, worker=worker, task=task):
					plane.supply_inputs(
						inputs, worker.owner, service.widths[task.layer - 1], stack, gather=service.gather_inputs
					)

				output.prepared = transport.prepare(
					worker.owner, packet, output_width=service.widths[task.layer], supply_inputs=supply
				)
			for output in outputs:
				assert output.prepared is not None
				output.pending = transport.launch(output.prepared)
		except BaseException as error:
			failure = error
		launched = sum(output.pending is not None for output in outputs)
		if launched > 1:
			service.counters.wave_launches += 1
			service.counters.wave_tasks_launched += launched
		# Always drain every successfully launched receive, even if a later launch
		# or a different model failed. Publish successful work so it is resumable.
		try:
			for output in outputs:
				if output.pending is not None:
					try:
						output.result = output.pending.result()
					except BaseException as error:
						if failure is None:
							failure = error
			for output in outputs:
				if output.result is not None:
					plane.publish_result(output.attempt, output.result)
					output.registered = True
					obsolete = service.catalog.publish(output.key)
					output.published = True
					service.counters.tasks += 1
					service.counters.rows += output.task.targets.numel()
					for key in obsolete:
						plane.retire(key)
		finally:
			for output in outputs:
				# I/O failure resets the actor. Otherwise cancellation precedes reuse.
				if output.prepared is not None and output.pending is None and not isinstance(failure, TransportFailure):
					with suppress(TransportFailure):
						transport.cancel(output.prepared)
				if not output.published:
					service.catalog.cancel(output.key)
					if output.registered or output.result is None:
						plane.retire(output.key)
					else:
						transport.retire(Location(output.worker.owner, output.result.handle))
						output.worker.bytes_used = max(
							0, output.worker.bytes_used - output.result.shape[0] * output.result.shape[1] * 4
						)
			release_workers()
	if failure is not None:
		raise failure
