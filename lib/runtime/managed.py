"""Host rank-zero storage and launch or join its configured worker world."""

import logging
import math
from multiprocessing.process import BaseProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union, cast

import torch
import torch.multiprocessing as mp
from dgl import DGLGraph
from torch import Tensor

from lib.dist.comm_manager import CommManager
from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.launch import Placement, WorkerSettings, join_world, resolve_devices, run_worker
from lib.runtime.policies import DeferredPolicyConfig, DispatchPolicy
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.storage.capacity import GIB, StorageNodeBudget

if TYPE_CHECKING:
	from lib.storage.pma import PMATopology
	from lib.storage.topology import ShardedTopology


class ManagedRuntime:
	"""Use one serving path for spawned and externally placed worker ranks.

	The storage process must be the sole owner of its default torch.distributed
	world. Graph/features are retained by reference and never passed to children.
	Spawned workers can be replaced while retaining acknowledged CPU replicas.
	Externally placed workers belong to their launcher, including restart policy.
	"""

	def __init__(
		self,
		graph: DGLGraph,
		features: Tensor,
		model: ModelSpec,
		devices: Sequence[str] = (),
		*,
		worker_bytes: int = 1024**3,
		storage_bytes: int = 2 * 1024**3,
		timeout: float = 30,
		max_retries: int = 2,
		profile_dir: Optional[str] = None,
		switches: Optional[RuntimeFeatures] = None,
		native_threads: int = 4,
		max_stale_versions: int = 2,
		idle_flush_seconds: float = 0.005,
		storage_partitions: int = 1,
		storage_directory: Optional[str] = None,
		storage_exported_base: Optional[str] = None,
		storage_overlay_bytes: int = 256 * 1024**2,
		storage_response_bytes: int = 256 * 1024**2,
		storage_node_names: Optional[Sequence[str]] = None,
		storage_node_capacity_bytes: int = 512 * GIB,
		storage_node_reserve_bytes: int = 64 * GIB,
		storage_endpoints: Optional[Sequence[str]] = None,
		storage_authkey: str = "neutronrt",
		delta_max_steps: int = 32,
		delta_anchor_bytes: Optional[int] = None,
		placement: Optional[Placement] = None,
		worker_threads: Optional[int] = None,
		history_bytes: Optional[int] = None,
		worker_feature_cache_bytes: int = 0,
		dispatch_mode: str = "round_robin",
		dispatch_locality_weight: float = 1.0,
		dispatch_load_weight: float = 1.0,
		deferred_mode: str = "version_window",
		deferred_window_seconds: Optional[float] = None,
		hotspot_degree_threshold: Optional[int] = None,
		hotspot_max_changes: Optional[int] = None,
		backend: str = "auto",
		rendezvous_prefix: Optional[str] = None,
		**service_options,
	) -> None:
		if (
			(not devices and (placement is None or placement.rank is None))
			or worker_bytes < 1
			or storage_bytes < 0
			or storage_node_capacity_bytes < 1
			or storage_node_reserve_bytes < 0
			or storage_node_reserve_bytes >= storage_node_capacity_bytes
			or timeout <= 0
			or max_retries < 0
			or type(worker_feature_cache_bytes) is not int
			or worker_feature_cache_bytes < 0
		):
			raise ValueError("Invalid managed worker devices, budgets, timeout, or retry limit")
		switches = switches or RuntimeFeatures()
		if "pipeline" in service_options:
			raise TypeError("Set pipeline through RuntimeFeatures")
		switches.require_supported(
			{
				"pipeline",
				"coalescing",
				"peer_fetch",
				"preemption",
				"recovery",
				"profiling",
				"native_analysis",
				"bounded_staleness",
				"gather_inputs",
				"concurrent_compute",
				"distributed_storage",
				"retain_aggregates",
				"aggregate_only",
				"incremental_aggregates",
				"row_refresh",
				"operator_incremental",
				"pma_storage",
				"mapped_host_reads",
				"worker_feature_cache",
			}
		)
		if worker_feature_cache_bytes and not switches.worker_feature_cache:
			raise ValueError("worker_feature_cache_bytes requires worker_feature_cache=true")
		if switches.worker_feature_cache and worker_feature_cache_bytes < 1:
			raise ValueError("worker_feature_cache=true requires a positive byte budget")
		if switches.worker_feature_cache:
			raise NotImplementedError(
				"worker_feature_cache is fail-closed: the current wire protocol sends complete feature rows "
				"before lookup; miss-only version/generation-certified transport is not implemented"
			)
		deferred_policy = DeferredPolicyConfig(
			mode=deferred_mode,
			window_seconds=deferred_window_seconds,
			hotspot_degree_threshold=hotspot_degree_threshold,
			hotspot_max_changes=hotspot_max_changes,
		)
		if not switches.bounded_staleness and deferred_policy.mode != "version_window":
			raise ValueError("Non-default deferred policies require bounded_staleness=true")
		dispatch_policy = service_options.pop(
			"dispatch_policy",
			DispatchPolicy(
				dispatch_mode,
				locality_weight=dispatch_locality_weight,
				load_weight=dispatch_load_weight,
			),
		)
		if switches.aggregate_only and not switches.retain_aggregates:
			raise ValueError("Aggregate-only storage requires retain_aggregates=true")
		policy = service_options.get("neighborhood_policy")
		if (
			policy is not None
			and not policy.reuse_intermediates
			and (switches.operator_incremental or switches.retain_aggregates or switches.bounded_staleness)
		):
			raise ValueError("Full/sampled recomputation cannot use incremental aggregates or deferred publication")
		if switches.operator_incremental and switches.retain_aggregates:
			raise ValueError("Operator incremental execution uses its own state; disable retain_aggregates")
		if switches.mapped_host_reads and not switches.operator_incremental:
			raise ValueError("Mapped historical reads require operator_incremental=true")
		if switches.incremental_aggregates and not switches.retain_aggregates:
			raise ValueError("Sparse updates require retain_aggregates=true")
		if switches.row_refresh and not switches.incremental_aggregates:
			raise ValueError("Row refresh requires incremental_aggregates=true")
		if delta_anchor_bytes is None:
			if placement is not None and placement.rank is not None:
				assert placement.world_size is not None
				workers = placement.world_size - 1
			else:
				workers = len(devices)
			# Anchors pin existing producer buffers; they do not allocate another
			# embedding matrix. Leave a quarter of retained capacity for new outputs.
			delta_anchor_bytes = worker_bytes * max(1, workers) * 3 // 4
		if (
			type(delta_max_steps) is not int
			or delta_max_steps < 1
			or type(delta_anchor_bytes) is not int
			or delta_anchor_bytes < 0
		):
			raise ValueError("Delta refresh steps must be positive and anchor bytes nonnegative")
		if switches.retain_aggregates and model.name != "sage":
			raise ValueError("Retained projected aggregates currently require mean GraphSAGE")
		if switches.retain_aggregates and (switches.bounded_staleness or switches.concurrent_compute):
			raise NotImplementedError("Retained aggregates require serial exact dispatch until wave/anchor integration")
		if (
			type(storage_partitions) is not int
			or storage_partitions < 1
			or min(storage_overlay_bytes, storage_response_bytes) < 0
		):
			raise ValueError("Invalid CPU storage partition count or byte budgets")
		if switches.native_analysis and switches.distributed_storage:
			raise NotImplementedError("Native frontier over sharded mutable topology is not implemented")
		if switches.distributed_storage and graph is not None and "nrt" in graph.edata:
			raise NotImplementedError("CPU topology actors do not yet retain typed edge attributes")
		if switches.pma_storage and (switches.distributed_storage or switches.native_analysis):
			raise ValueError("PMA storage currently uses local CPU analysis without topology actors")
		if storage_partitions != 1 and not switches.distributed_storage:
			raise ValueError("Multiple CPU storage partitions require distributed_storage=true")
		if type(native_threads) is not int or native_threads < 1:
			raise ValueError("Native thread count must be a positive integer")
		if (
			type(max_stale_versions) is not int
			or max_stale_versions < 0
			or not math.isfinite(idle_flush_seconds)
			or idle_flush_seconds <= 0
		):
			raise ValueError("Invalid shared staleness budget or idle flush interval")
		if torch.distributed.is_initialized():
			raise RuntimeError("Managed runtime requires ownership of the default communication world")
		self.devices = tuple(torch.device(device) for device in resolve_devices(devices)) if devices else ()
		self.placement = placement or Placement()
		if self.placement.rank not in (None, 0):
			raise ValueError("The storage runtime belongs to rank zero; other ranks run workers")
		self.worker_settings = WorkerSettings(
			switches=switches,
			threads=worker_threads if worker_threads is not None else torch.get_num_threads(),
			backend=backend,
			timeout=timeout,
			profile_dir=profile_dir,
			history_bytes=256 * 1024**2 if history_bytes is None else history_bytes,
			refresh_steps=delta_max_steps,
			feature_cache_bytes=worker_feature_cache_bytes,
			rendezvous_prefix=rendezvous_prefix,
		)
		self.model = model
		self.switches = switches
		self.profile_dir = profile_dir
		self.worker_bytes, self.timeout = worker_bytes, timeout
		self.processes: Dict[int, BaseProcess] = {}
		self.comm: Optional[CommManager] = None
		self._lock = RLock()
		self._closed = False
		self._world_sequence = 0
		self.reset_count = 0
		self._folder = TemporaryDirectory(prefix="neutronrt-workers-")
		storage_names = tuple(storage_node_names or (f"local-{index}" for index in range(storage_partitions)))
		if len(storage_names) != storage_partitions or len(set(storage_names)) != len(storage_names):
			raise ValueError("Storage node names must be distinct and match storage_partitions")
		if storage_endpoints is not None and len(storage_endpoints) != storage_partitions:
			raise ValueError("One storage endpoint is required per storage partition")
		self.storage_budgets = tuple(
			StorageNodeBudget(name, storage_node_capacity_bytes, storage_node_reserve_bytes) for name in storage_names
		)
		self.plane: Optional[RemoteEmbeddingPlane] = None
		self.topology: Optional[Union["PMATopology", "ShardedTopology"]] = None
		try:
			if switches.pma_storage:
				from lib.storage.pma import PMATopology

				self.topology = PMATopology(graph)
				graph = cast(DGLGraph, self.topology)
			if switches.distributed_storage:
				from lib.storage.topology import ShardedTopology

				self.topology = ShardedTopology(
					graph,
					storage_partitions,
					directory=storage_directory,
					overlay_bytes=storage_overlay_bytes,
					response_bytes=storage_response_bytes,
					timeout=timeout,
					profiling=switches.profiling,
					storage_budgets=self.storage_budgets,
					storage_endpoints=tuple(storage_endpoints) if storage_endpoints is not None else None,
					storage_authkey=storage_authkey,
					exported_base=storage_exported_base,
				)
				graph = cast(DGLGraph, self.topology)
				service_options["frontier_expander"] = self.topology.expand
			if switches.native_analysis:
				from lib.graph.native_frontier import NativeFrontier

				service_options["frontier_expander"] = NativeFrontier(graph, threads=native_threads)
			endpoints = [
				RemoteEndpoint(Owner(rank, 1), device, worker_bytes)
				for rank, device in enumerate(self.devices, start=1)
			]
			transport = self._start(endpoints)
			plane = RemoteEmbeddingPlane(transport, storage_bytes=storage_bytes)
			self.plane = plane
			# Keep lifecycle policy separate from ordinary distributed execution.
			from lib.runtime.recovery import ResilientEmbeddingService

			service_options.setdefault("source_degree_sensitive", model.name == "gcn")
			service_options["pipeline"] = switches.pipeline
			service_options["preemption_enabled"] = switches.preemption
			service_options["dispatch_policy"] = dispatch_policy
			service_options["worker_feature_cache_bytes"] = worker_feature_cache_bytes
			self.service = ResilientEmbeddingService(
				graph,
				features,
				model.widths,
				plane,
				self,
				peer_fetch=switches.peer_fetch,
				gather_inputs=switches.gather_inputs,
				concurrent_compute=switches.concurrent_compute,
				retain_aggregates=switches.retain_aggregates,
				aggregate_only=switches.aggregate_only,
				incremental_aggregates=switches.incremental_aggregates,
				row_refresh=switches.row_refresh,
				delta_max_steps=delta_max_steps,
				delta_anchor_bytes=delta_anchor_bytes,
				max_retries=max_retries if switches.recovery and self.placement.rank is None else 0,
				**service_options,
			)
			if switches.bounded_staleness:
				from lib.graph.deferred import DeferredUpdates

				DeferredUpdates(
					self.service,
					max_versions=max_stale_versions,
					idle_seconds=idle_flush_seconds,
					policy=deferred_policy,
				)
		except BaseException:
			self._abort_world()
			if self.plane is not None:
				self.plane.close()
			if self.topology is not None:
				self.topology.close()
			self._folder.cleanup()
			raise

	@torch.inference_mode(False)
	def _start(self, endpoints: Sequence[RemoteEndpoint]) -> OwnedTransport:
		# Collective setup writes from backend threads. Recovery can be invoked
		# inside a model task's inference_mode; its setup buffers must be normal.
		self._world_sequence += 1
		placement = self.placement
		if placement.rank is None:
			rendezvous = f"file://{Path(self._folder.name) / str(self._world_sequence)}"
			world_size = len(endpoints) + 1
			context = mp.get_context("spawn")
			for endpoint in endpoints:
				process = context.Process(
					target=run_worker,
					args=(self.model, endpoint, world_size, rendezvous, self.worker_settings),
					name=f"neutronrt-worker-{endpoint.owner.rank}-g{endpoint.owner.generation}",
				)
				process.start()
				self.processes[endpoint.owner.rank] = process
		else:
			assert placement.world_size is not None
			rendezvous, world_size = placement.init_method, placement.world_size
		self.comm, control, endpoints, _ = join_world(
			self.model, 0, world_size, torch.device("cpu"), 1, 0, rendezvous, self.worker_settings
		)
		torch.distributed.barrier()
		return OwnedTransport(self.comm, control, endpoints, coalesced=self.switches.coalescing)

	def healthy(self) -> bool:
		return (
			not self._closed
			and self.comm is not None
			and (self.placement.rank is not None or bool(self.processes))
			and all(process.is_alive() for process in self.processes.values())
		)

	def _join_workers(self, seconds: float) -> None:
		deadline = monotonic() + seconds
		for process in self.processes.values():
			process.join(max(0, deadline - monotonic()))

	def _abort_world(self) -> None:
		for process in self.processes.values():
			if process.is_alive():
				process.terminate()
		self._join_workers(5)
		for process in self.processes.values():
			if process.is_alive():
				process.kill()
		self._join_workers(5)
		self.processes.clear()
		if self.comm is not None:
			self.comm.abort()
			self.comm = None

	def reset(self) -> Sequence[RemoteEndpoint]:
		"""Caller holds the service graph fence; old writebacks remain fenced."""
		with self._lock:
			if self.placement.rank is not None:
				raise RuntimeError("Worker replacement belongs to the external process launcher")
			if self._closed or self.plane is None:
				raise RuntimeError("Managed runtime is closed or not initialized")
			plane = self.plane
			logging.warning("Resetting all managed worker communicators after compute/transport failure")
			for owner in tuple(plane.remotes):
				plane.lose_owner(owner)
			self._abort_world()
			endpoints = [
				RemoteEndpoint(plane.directory.register_owner(rank), device, self.worker_bytes)
				for rank, device in enumerate(self.devices, start=1)
			]
			# Register replacement owners even if startup fails, so another reset
			# can invalidate those generations before trying the next rendezvous.
			plane.remotes = {endpoint.owner: endpoint for endpoint in endpoints}
			try:
				transport = self._start(endpoints)
			except BaseException:
				self._abort_world()
				raise
			plane.transport = transport
			self.reset_count += 1
			return endpoints

	def close(self) -> None:
		"""Drain client admission/query helpers before closing their runtime."""
		if self._closed:
			return
		try:
			# Drain before acquiring the world lock: a timer can hold the service
			# fence while recovery needs that lock. Keep workers healthy until done.
			if self.service.deferred is not None:
				self.service.deferred.close()
		finally:
			self._close_workers()

	def _close_workers(self) -> None:
		with self._lock:
			if self._closed:
				return
			healthy = self.healthy()
			self._closed = True
			try:
				if not healthy:
					self._abort_world()
				if self.plane is not None:
					self.plane.close()
				if healthy and self.plane is not None and self.comm is not None:
					self.plane.transport.stop()
					self.comm.destroy()
					self._join_workers(5)
			finally:
				self._abort_world()
				if self.topology is not None:
					self.topology.close()
				self._folder.cleanup()

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
