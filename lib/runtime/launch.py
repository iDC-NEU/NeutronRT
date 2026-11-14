"""Resolve process placement and join the same worker protocol on any host."""

import os
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from lib.dist.comm_manager import CommManager
from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.transport import RemoteEndpoint
from lib.runtime.worker import OwnedWorker


def resolve_devices(devices: Optional[Sequence[str]] = None) -> tuple[str, ...]:
	values = devices or tuple(f"cuda:{index}" for index in range(torch.cuda.device_count())) or ("cpu",)
	result = []
	for value in values:
		device = torch.device(value)
		if device.type not in ("cpu", "cuda"):
			raise ValueError(f"Unsupported worker device: {value}")
		if device.type == "cuda":
			device = torch.device("cuda", 0 if device.index is None else device.index)
			if device.index >= torch.cuda.device_count():
				raise ValueError(f"Worker device {device} is outside the visible CUDA devices")
		result.append(str(device))
	return tuple(result)


@dataclass(frozen=True)
class Placement:
	rank: Optional[int] = None
	world_size: Optional[int] = None
	local_rank: int = 0
	init_method: str = "env://"

	@classmethod
	def resolve(
		cls,
		*,
		rank=None,
		world_size=None,
		local_rank=None,
		init_method=None,
		environ: Optional[Mapping[str, str]] = None,
	):
		env = os.environ if environ is None else environ
		rank = rank if rank is not None else int(env["RANK"]) if "RANK" in env else None
		world_size = (
			world_size
			if world_size is not None
			else int(env["WORLD_SIZE"])
			if rank is not None and "WORLD_SIZE" in env
			else None
		)
		local_rank = local_rank if local_rank is not None else int(env.get("LOCAL_RANK", 0))
		if (
			(rank is None) != (world_size is None)
			or world_size is not None
			and (world_size < 2 or not 0 <= rank < world_size)
			or local_rank < 0
		):
			raise ValueError("Placement requires rank in [0, world_size), world_size >= 2 and a nonnegative local rank")
		return cls(rank, world_size, local_rank, init_method or "env://")

	def worker_device(self, devices=None, device=None):
		if device is not None and device != "cuda":
			return resolve_devices([device])[0]
		available = resolve_devices(devices)
		if device == "cuda" and available == ("cpu",):
			raise ValueError("No visible CUDA device for this worker")
		# Rank zero occupies a CPU slot only on its own host.
		index = self.local_rank - (1 if self.rank == self.local_rank else 0)
		if available == ("cpu",):
			return "cpu"
		if not 0 <= index < len(available):
			raise ValueError("Local worker rank exceeds configured devices; pass --device or --workers")
		return available[index]


@dataclass(frozen=True)
class WorkerSettings:
	switches: RuntimeFeatures = field(default_factory=RuntimeFeatures)
	threads: int = 1
	backend: str = "auto"
	timeout: float = 30
	profile_dir: Optional[str] = None
	history_bytes: int = 256 * 1024**2
	refresh_steps: int = 32
	checkpoint: Optional[str] = None
	rendezvous_prefix: Optional[str] = None

	def __post_init__(self):
		if self.threads < 1 or self.timeout <= 0 or self.history_bytes < 0 or self.refresh_steps < 1:
			raise ValueError("Invalid worker threads, timeout or history budget")
		if self.backend not in ("auto", "gloo"):
			raise ValueError("CPU storage requires a CPU-capable control backend: auto or gloo")


def join_world(
	spec: Optional[ModelSpec], rank, world_size, device, generation, max_bytes, rendezvous, settings: WorkerSettings
):
	comm_type = CommManager
	if settings.switches.profiling:
		from lib.profiling.communication import ProfiledCommManager

		comm_type = ProfiledCommManager
	manager = comm_type.initialize(
		world_size,
		rank,
		device=device,
		backend=settings.backend,
		init_method=rendezvous,
		timeout=timedelta(seconds=settings.timeout),
		**({"rendezvous_prefix": settings.rendezvous_prefix} if settings.rendezvous_prefix is not None else {}),
	)
	try:
		model_info = [spec if rank == 0 else None]
		dist.broadcast_object_list(model_info, src=0)
		if spec is None:
			spec = model_info[0]
		if spec is None:
			raise ValueError("The storage rank did not broadcast a model specification")
		identity = (
			spec.name,
			spec.widths,
			spec.heads,
			spec.relations,
			spec.pseudo_dim,
			spec.kernels,
			spec.seed,
			settings.switches.mask,
		)
		peers: list[Any] = [None] * world_size
		dist.all_gather_object(peers, (str(device), generation, max_bytes, identity))
		if any(peer[3] != identity for peer in peers):
			raise ValueError("Ranks disagree on model configuration or runtime feature switches")
		control = manager.create_epoch(range(world_size), timeout=timedelta(seconds=settings.timeout))
		endpoints = [
			RemoteEndpoint(Owner(peer_rank, peer[1]), torch.device(peer[0]), peer[2])
			for peer_rank, peer in enumerate(peers)
			if peer_rank
		]
		return manager, control, endpoints, spec
	except BaseException:
		manager.abort()
		raise


def run_worker(
	spec: Optional[ModelSpec], endpoint: RemoteEndpoint, world_size: int, rendezvous: str, settings: WorkerSettings
):
	torch.set_num_threads(settings.threads)
	manager, control, _, spec = join_world(
		spec,
		endpoint.owner.rank,
		world_size,
		endpoint.device,
		endpoint.owner.generation,
		endpoint.max_bytes,
		rendezvous,
		settings,
	)
	if settings.checkpoint is not None:
		spec = replace(spec, checkpoint=settings.checkpoint)
	model = spec.build().to(endpoint.device)
	dist.barrier()
	worker_type: type[OwnedWorker] = OwnedWorker
	kwargs = {}
	if settings.switches.profiling:
		from lib.profiling.worker import ProfiledWorker

		worker_type = ProfiledWorker
		kwargs["profile_dir"] = settings.profile_dir
	worker_type(
		manager,
		model,
		endpoint.owner,
		max_bytes=endpoint.max_bytes,
		control_epoch=control,
		coalesced=settings.switches.coalescing,
		gather_inputs=settings.switches.gather_inputs,
		operator_incremental=settings.switches.operator_incremental,
		mapped_host_reads=settings.switches.mapped_host_reads,
		history_bytes=settings.history_bytes,
		refresh_steps=settings.refresh_steps,
		**kwargs,
	).run()
