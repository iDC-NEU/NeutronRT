"""Profile the actual coordinator/worker pipeline without changing its schedule."""

from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from threading import RLock
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from lib.graph.methods import method_from_switches
from lib.graph.serving import UpdateResult
from lib.profiling.communication import ProfiledCommManager
from lib.profiling.control import start_workers, stop_workers
from lib.profiling.instrumentation import (
	counter_delta,
	counter_snapshot,
	grouped_counter_delta,
	profile_snapshot,
	profile_window,
)
from lib.profiling.network import NetworkSnapshot
from lib.profiling.network import delta as network_delta
from lib.profiling.network import snapshot as network_snapshot
from lib.profiling.recorder import MemorySampler
from lib.profiling.residency import retained_state
from lib.profiling.storage import start as start_storage
from lib.profiling.storage import stop as stop_storage
from lib.runtime.data_plane import RemoteEmbeddingPlane

if TYPE_CHECKING:
	from lib.runtime.managed import ManagedRuntime


def payload_total(reports: List[dict], direction: str) -> int:
	return sum(
		value
		for report in reports
		for name, value in report["counters"].items()
		if name.startswith(f"p2p.{direction}.") and name.endswith(".payload_bytes")
	)


class RuntimeProfiler:
	"""Measure complete windows; trace capture is intended for a separate pass.

	Phase totals overlap, CUDA events measure the model stream, and P2P counters
	measure tensor payloads. Neither timing totals nor endpoint byte counts should
	be blindly added. A worker-world reset invalidates normal-throughput comparison.
	"""

	def __init__(
		self,
		runtime: "ManagedRuntime",
		*,
		network_interfaces: Optional[List[str]] = None,
		admission=None,
	) -> None:
		if not runtime.switches.profiling or not isinstance(runtime.comm, ProfiledCommManager) or runtime.plane is None:
			raise ValueError("Runtime profiling must be enabled before starting the worker world")
		self.runtime = runtime
		self.comm = runtime.comm
		self.recorder = runtime.comm.recorder
		self.transport = runtime.plane.transport
		self._lock = RLock()
		self._patched: List[Tuple[object, str, Any, bool]] = []
		self._rss: Optional[MemorySampler] = None
		self._storage_before: Optional[dict] = None
		self._storage_samplers: dict = {}
		self._before: dict = {}
		self._network_interfaces = tuple(network_interfaces) if network_interfaces is not None else None
		self._network_before: Optional[NetworkSnapshot] = None
		self._reset_count = 0
		self.admission = admission
		self._admission_before: Optional[dict] = None
		self._policy_before: Optional[dict] = None
		self._first_update = self._last_update = 0
		self._install()

	def attach_admission(self, admission) -> None:
		"""Attach an optional UpdateAdmission observer before the next window."""
		if self.recorder.active:
			raise RuntimeError("Cannot attach admission while a profile window is active")
		if admission is not None and getattr(admission, "service", None) is not self.runtime.service:
			raise ValueError("Admission belongs to another embedding service")
		self.admission = admission

	def _wrap(self, target, name: str, phase: str, *, update: bool = False, generator: bool = False) -> None:
		original: Callable[..., Any] = getattr(target, name)
		had_instance = name in vars(target)
		previous = vars(target).get(name)
		self._patched.append((target, name, previous, had_instance))
		if generator:

			@wraps(original)
			def iterate(*args, **kwargs):
				iterator = iter(original(*args, **kwargs))
				while True:
					with self.recorder.span(phase):
						try:
							item = next(iterator)
						except StopIteration:
							return
					yield item

			setattr(target, name, iterate)
			return

		@wraps(original)
		def invoke(*args, **kwargs):
			if not self.recorder.active:
				return original(*args, **kwargs)
			# A deferred future can wake its caller inside the final publication.
			# Finish observer spans before releasing the same fence stop() acquires.
			with self.runtime.service._fence if update else nullcontext():
				start = perf_counter_ns()
				result = None
				try:
					with self.recorder.span(phase):
						result = original(*args, **kwargs)
					if update and isinstance(result, UpdateResult):
						self.recorder.count("pipeline.prepare_ns", round(result.preparation_seconds * 1e9))
						self.recorder.count("pipeline.compute_thread_ns", round(result.compute_seconds * 1e9))
						self.recorder.count("pipeline.exposed_wait_ns", round(result.wait_seconds * 1e9))
					return result
				finally:
					if update and result is not None:
						with self._lock:
							self._first_update = min(self._first_update or start, start)
							self._last_update = max(self._last_update, perf_counter_ns())

		setattr(target, name, invoke)

	def _install(self) -> None:
		service = self.runtime.service
		for name, label in (("change_features", "features"), ("insert_edges", "insert"), ("delete_edges", "delete")):
			self._wrap(service, name, f"root.update.{label}", update=True)
		if service.deferred is not None:
			self._wrap(service.deferred, "submit", "root.stage_update", update=True)
			self._wrap(service.deferred, "flush", "root.publish_merged_update", update=True)
		for target, name, phase in (
			(service.graph, "add_edges", "root.graph_insert"),
			(service.graph, "remove_edges", "root.graph_delete"),
			(service.graph, "delete_pairs", "root.graph_delete"),
			(service.graph, "expand", "root.storage_expand"),
			(service.graph, "compile_block", "root.storage_block"),
			(service.graph, "sample_block", "root.storage_sampled_block"),
			(service.graph, "create_formats_", "root.graph_formats"),
			(service.affected, "block", "root.block_compile"),
			(service.catalog, "publish", "root.publish"),
			(service.catalog, "selections", "root.resolve_rows"),
			(self.transport, "compute", "root.task_dispatch_and_ready"),
			(self.transport, "compute_with_aggregate", "root.task_dispatch_and_ready"),
			(self.transport, "compute_from_aggregates", "root.task_dispatch_and_ready"),
			(self.transport, "compute_delta", "root.delta_dispatch_and_ready"),
			(self.transport, "reconstruct", "root.reconstruction_dispatch_and_ready"),
			(self.transport, "prepare", "root.input_preparation"),
			(self.transport, "launch", "root.concurrent_launch"),
			(service, "_run_wave", "root.concurrent_wave"),
			(service, "_run", "root.recomputation"),
			(service, "_compute_layer", "root.layer_compute"),
			(self.transport, "read", "root.selected_read_post"),
			(self.runtime.plane, "supply_inputs", "root.input_routing"),
			(self.runtime.plane, "gather_rows", "root.query_gather"),
		):
			if hasattr(target, name):
				self._wrap(target, name, phase)
		# Sizing rejections are internal to streamed planning. An exception that
		# escapes the iterator is a failed request; a successful split is not.
		self._wrap(service.planner, "plans", "root.dependency_plan", generator=True)
		aggregates = getattr(service, "aggregate_catalog", None)
		if aggregates is not None:
			self._wrap(aggregates, "publish", "root.publish_aggregate")
			self._wrap(aggregates, "selections", "root.resolve_aggregate_rows")
		self._wrap(service.affected, "analyze", "root.affected_analysis", generator=True)

	def start(self, *, trace: bool = False, trace_task_limit: int = 100) -> None:
		service = self.runtime.service
		with service._fence:
			if self.recorder.active:
				raise RuntimeError("Runtime measurement window is already active")
			if not self._publication_state()["service_drained"]:
				raise RuntimeError("Finish pending update publication before starting a measurement window")
			if trace and self.runtime.profile_dir is None:
				raise ValueError("Trace capture requires a profile directory")
			self._first_update = self._last_update = 0
			self._before = counter_snapshot(service.counters)
			self._policy_before = (
				profile_snapshot(service.profile()) if callable(getattr(service, "profile", None)) else None
			)
			self._admission_before = self.admission.profile_state() if self.admission is not None else None
			if self._admission_before is not None and self._admission_before["pending"]:
				raise RuntimeError("Finish pending admission work before starting a measurement window")
			self._reset_count = self.runtime.reset_count
			start_workers(self.transport, trace=trace, trace_task_limit=trace_task_limit)
			self._storage_before, self._storage_samplers = start_storage(getattr(self.runtime, "topology", None))
			self._network_before = network_snapshot(self._network_interfaces)
			self._rss = MemorySampler()
			self._rss.start()
			path = None
			if trace:
				assert self.runtime.profile_dir is not None
				path = Path(self.runtime.profile_dir) / "coordinator.phases.json"
			self.recorder.start(path)

	def _publication_state(self) -> dict:
		"""The caller holds the service fence; queued RPC admission is external."""
		service = self.runtime.service
		pending = service.deferred.pending if service.deferred is not None else 0
		incomplete = service._pending_update is not None
		result = {
			"pending_deferred_updates": pending,
			"update_incomplete": incomplete,
			"service_drained": pending == 0 and not incomplete,
		}
		if self.admission is not None:
			state = self.admission.profile_state()
			result["admission"] = state
			result["admission_drained"] = state["pending"] == 0
		return result

	def stop(self) -> dict:
		service = self.runtime.service
		with service._fence:
			if self._rss is None:
				raise RuntimeError("No active runtime measurement window")
			publication = self._publication_state()
			root = self.recorder.stop()
			root["rss"] = self._rss.stop()
			self._rss = None
			if self._network_before is None:
				raise RuntimeError("Network measurement window was not initialized")
			network = network_delta(self._network_before, network_snapshot(self._network_interfaces))
			self._network_before = None
			reset = self.runtime.reset_count != self._reset_count
			workers = [] if reset else stop_workers(self.transport)
			storage = stop_storage(
				getattr(self.runtime, "topology", None), self._storage_before, self._storage_samplers
			)
			self._storage_before, self._storage_samplers = None, {}
			after = counter_snapshot(service.counters)
			policy_after = profile_snapshot(service.profile()) if callable(getattr(service, "profile", None)) else None
			policy_window = (
				profile_window(self._policy_before, policy_after)
				if self._policy_before is not None and isinstance(policy_after, dict)
				else None
			)
			delta, counter_resets = counter_delta(self._before, after)
			engine_window = (self._last_update - self._first_update) / 1e9 if self._first_update else 0.0
			busy_seconds = delta.get("update_seconds", 0) + delta.get("staging_seconds", 0)
			all_reports = [root, *workers]
			sent, received = payload_total(all_reports, "send"), payload_total(all_reports, "recv")
			admission_drained = publication.get("admission_drained", True)
			storage_sampling_complete = storage is None or storage.get("sampling_complete", True)
			admission_window = None
			if self._admission_before is not None and self.admission is not None:
				current_admission = self.admission.profile_state()
				admission_window = {
					name: current_admission[name] - self._admission_before[name]
					for name in ("admitted", "completed", "failed")
				}
				admission_window["pending_at_stop"] = current_admission["pending"]
			valid = (
				publication["service_drained"]
				and admission_drained
				and (admission_window is None or admission_window["failed"] == 0)
				and storage_sampling_complete
				and (policy_window is None or not policy_window["counter_resets"])
				and not reset
				and sent == received
				and all(report["open_spans"] == 0 and report["series_dropped"] == 0 for report in all_reports)
				and all(timing["failures"] == 0 for report in all_reports for timing in report["timings"].values())
			)
			sparse = getattr(service, "_delta", None)
			return {
				"schema_version": 2,
				"method": method_from_switches(self.runtime.switches.as_dict(), policy=service.planner.policy),
				"neighborhood_policy": {
					"fanout": service.planner.policy.fanout,
					"reuse_intermediates": service.planner.policy.reuse_intermediates,
					"seed": service.planner.policy.seed,
				},
				"valid": valid,
				"publication": publication,
				"admission_window": admission_window,
				"policy_window": policy_window,
				"instrumentation": {
					"counter_source": "EmbeddingService.counters",
					"counter_deltas": delta,
					"counter_resets": counter_resets,
					"groups": grouped_counter_delta(delta),
					"reset_semantics": "A counter that decreased is reported as zero and named in counter_resets.",
				},
				"worker_world_reset": reset,
				"traffic_balanced": sent == received,
				"trace_pass": any(report["trace_path"] for report in all_reports),
				"coordinator": root,
				"workers": workers,
				"storage": storage,
				"delta_state": {
					"row_refresh": sparse.state.row_refresh,
					"depth_tensor_bytes": sparse.state.metadata_bytes,
					"scope": "End-of-window integer tensor storage; excludes Python/allocator overhead",
				}
				if sparse is not None
				else None,
				"retained_state": retained_state(self.runtime.plane)
				if isinstance(self.runtime.plane, RemoteEmbeddingPlane)
				else None,
				"engine": {
					**delta,
					"completion_window_seconds": engine_window,
					"counter_resets": counter_resets,
					"completed_changes_per_second": delta["input_changes"] / engine_window if engine_window else None,
					"busy_changes_per_second": delta["input_changes"] / busy_seconds if busy_seconds else None,
				},
				"p2p": {
					"send_payload_bytes": sent,
					"recv_payload_bytes": received,
					"send_operations": sum(
						value
						for report in all_reports
						for name, value in report["counters"].items()
						if name.startswith("p2p.send.") and name.endswith(".operations")
					),
				},
				"network": network,
			}

	def close(self) -> None:
		try:
			if self.recorder.active:
				self.stop()
		finally:
			for sampler in self._storage_samplers.values():
				sampler.stop()
			self._storage_samplers = {}
			self._network_before = None
			for target, name, previous, had_instance in reversed(self._patched):
				if had_instance:
					setattr(target, name, previous)
				else:
					delattr(target, name)
			self._patched.clear()

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
