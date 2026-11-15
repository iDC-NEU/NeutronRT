"""Serve inference with rank-zero storage and workers placed by the process launcher."""

import json
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from threading import Event
from typing import Annotated

import grpc
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.methods import (
	METHODS,
	NRT_INC,
	method_from_switches,
	neighborhood_policy,
)
from lib.graph.ownership import Owner
from lib.models import MODEL_NAMES
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.rpc.inference_pb2_grpc import add_OwnedInferenceServicer_to_server
from lib.rpc.limits import message_options
from lib.rpc.service import OwnedServicer
from lib.runtime.config import RuntimeFeatures
from lib.runtime.launch import Placement, WorkerSettings, resolve_devices, run_worker
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import RemoteEndpoint
from lib.storage.capacity import GIB


def _export_profile(profiler, path, switches):
	report = profiler.stop()
	report["switches"] = switches.as_dict()
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(report, indent=2) + "\n")
	logging.info("Runtime profile written to %s; valid=%s", path, report["valid"])


def write_ready(path, listen, port, session):
	"""Publish only after binding; validation clients verify this PID and session."""
	address = listen if listen.startswith("unix:") else f"{listen.rsplit(':', 1)[0]}:{port}"
	value = {"pid": os.getpid(), "address": address, "session": session}
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
	temporary.write_text(json.dumps(value) + "\n")
	temporary.replace(path)
	return value


BackendChoice = StrEnum("BackendChoice", {str(value).replace("-", "_").upper(): value for value in ["auto", "gloo"]})  # type: ignore[misc]  # Mypy requires literal Enum members.
ModelChoice = StrEnum("ModelChoice", {str(value).replace("-", "_").upper(): value for value in MODEL_NAMES})  # type: ignore[misc]  # Mypy requires literal Enum members.
MethodChoice = StrEnum("MethodChoice", {str(value).replace("-", "_").upper(): value for value in METHODS})  # type: ignore[misc]  # Mypy requires literal Enum members.


_DEFAULT_BACKEND = BackendChoice("auto")


@dataclass
class Options:
	data: Path | None
	add_inverse_edges: bool
	workers: list[str] | None
	device: str | None
	rank: int | None
	world_size: int | None
	local_rank: int | None
	init_method: str | None
	rendezvous_prefix: str | None
	backend: str
	env: Path | None
	model: str | None
	method: str | None
	fanout: int
	sampling_seed: int
	checkpoint: Path | None
	hidden: int | None
	classes: int | None
	layers: int | None
	heads: int | None
	relations: int | None
	pseudo_dim: int | None
	kernels: int | None
	seed: int | None
	threads: int | None
	storage_partitions: int
	storage_node_names: list[str] | None
	storage_node_capacity_bytes: int
	storage_node_reserve_bytes: int
	storage_endpoints: list[str] | None
	storage_authkey: str
	network_interfaces: list[str] | None
	native_threads: int
	max_stale_versions: int
	idle_flush_seconds: float
	rpc_threads: int
	rpc_timing: bool
	listen: str | None
	ready_file: Path | None
	worker_bytes: int
	topology_root: Path | None
	storage_bytes: int
	storage_overlay_bytes: int
	storage_response_bytes: int
	batch_size: int
	max_plan_edges: int | None
	delta_max_steps: int
	history_bytes: int
	delta_anchor_bytes: int | None
	max_pending: int
	max_rpc_bytes: int
	max_tickets: int
	ticket_retention: float
	timeout: float
	retries: int
	serial: bool
	features: Path | None
	profile_report: Path | None
	pipeline: bool | None
	coalescing: bool | None
	peer_fetch: bool | None
	gather_inputs: bool | None
	preemption: bool | None
	recovery: bool | None
	profiling: bool | None
	native_analysis: bool | None
	incremental_aggregates: bool | None
	bounded_staleness: bool | None
	worker_feature_cache: bool | None
	alltoall: bool | None
	elastic_groups: bool | None
	distributed_storage: bool | None
	concurrent_compute: bool | None
	retain_aggregates: bool | None
	aggregate_only: bool | None
	row_refresh: bool | None
	operator_incremental: bool | None
	pma_storage: bool | None
	mapped_host_reads: bool | None


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[
		Path | None, typer.Option("--data", help="Prepared graph, DGL binary, or OGB raw directory (storage rank only)")
	] = None,
	add_inverse_edges: Annotated[
		bool, typer.Option("--add-inverse-edges", help="Include reverse edges, including self loops")
	] = False,
	workers: Annotated[
		list[str] | None,
		typer.Option("--workers", help="Worker devices on this host; otherwise discover visible devices"),
	] = None,
	device: Annotated[
		str | None, typer.Option("--device", help="Device for this process, overriding automatic worker placement")
	] = None,
	rank: Annotated[int | None, typer.Option("--rank")] = None,
	world_size: Annotated[int | None, typer.Option("--world-size")] = None,
	local_rank: Annotated[int | None, typer.Option("--local-rank", "--local_rank")] = None,
	init_method: Annotated[
		str | None, typer.Option("--init-method", help="Process rendezvous URL; defaults to launcher environment")
	] = None,
	rendezvous_prefix: Annotated[
		str | None,
		typer.Option("--rendezvous-prefix", help="Namespace for a world sharing a persistent rendezvous store"),
	] = None,
	backend: Annotated[BackendChoice, typer.Option("--backend")] = _DEFAULT_BACKEND,
	env: Annotated[Path | None, typer.Option("--env", help="Optional topology/network environment JSON")] = None,
	model: Annotated[ModelChoice | None, typer.Option("--model")] = None,
	method: Annotated[MethodChoice | None, typer.Option("--method")] = None,
	fanout: Annotated[int, typer.Option("--fanout", min=1, help="Neighbors per node at every layer of RTEC-NS")] = 5,
	sampling_seed: Annotated[int, typer.Option("--sampling-seed")] = 0,
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	hidden: Annotated[int | None, typer.Option("--hidden")] = None,
	classes: Annotated[int | None, typer.Option("--classes")] = None,
	layers: Annotated[int | None, typer.Option("--layers")] = None,
	heads: Annotated[int | None, typer.Option("--heads")] = None,
	relations: Annotated[int | None, typer.Option("--relations")] = None,
	pseudo_dim: Annotated[int | None, typer.Option("--pseudo-dim")] = None,
	kernels: Annotated[int | None, typer.Option("--kernels")] = None,
	seed: Annotated[int | None, typer.Option("--seed")] = None,
	threads: Annotated[
		int | None,
		typer.Option("--threads", help="Worker intra-op threads; defaults to the PyTorch/environment setting"),
	] = None,
	storage_partitions: Annotated[int, typer.Option("--storage-partitions")] = 1,
	storage_node_names: Annotated[
		list[str] | None,
		typer.Option("--storage-node", help="Storage node label; repeat once per storage partition"),
	] = None,
	storage_node_capacity_bytes: Annotated[
		int, typer.Option("--storage-node-capacity-bytes", help="Physical capacity for each storage node")
	] = 512 * GIB,
	storage_node_reserve_bytes: Annotated[
		int, typer.Option("--storage-node-reserve-bytes", help="Memory reserved for common server use")
	] = 64 * GIB,
	storage_endpoints: Annotated[
		list[str] | None,
		typer.Option("--storage-endpoint", help="Remote storage HOST:PORT; repeat once per storage partition"),
	] = None,
	storage_authkey: Annotated[str, typer.Option("--storage-authkey")] = "neutronrt",
	network_interfaces: Annotated[
		list[str] | None,
		typer.Option("--network-interface", help="OS interface to sample; repeat to select several"),
	] = None,
	native_threads: Annotated[int, typer.Option("--native-threads")] = 4,
	max_stale_versions: Annotated[int, typer.Option("--max-stale-versions")] = 2,
	idle_flush_seconds: Annotated[float, typer.Option("--idle-flush-seconds")] = 0.005,
	rpc_threads: Annotated[int, typer.Option("--rpc-threads")] = 8,
	rpc_timing: Annotated[
		bool, typer.Option("--rpc-timing", help="Expose handler duration in gRPC trailing metadata")
	] = False,
	listen: Annotated[
		str | None, typer.Option("--listen", help="RPC bind address; defaults to MASTER_ADDR and MASTER_RPC_PORT")
	] = None,
	ready_file: Annotated[
		Path | None, typer.Option("--ready-file", help="Optional atomic startup PID/address/session notification")
	] = None,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 1024**3,
	topology_root: Annotated[
		Path | None,
		typer.Option("--topology-root", help="Storage working directory; otherwise use the system temporary directory"),
	] = None,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 2 * 1024**3,
	storage_overlay_bytes: Annotated[
		int,
		typer.Option(
			"--storage-overlay-bytes", min=0, help="Changed adjacency bytes per direction and storage partition"
		),
	] = 256 * 1024**2,
	storage_response_bytes: Annotated[
		int, typer.Option("--storage-response-bytes", min=0, help="Maximum selected adjacency bytes per read operation")
	] = 256 * 1024**2,
	batch_size: Annotated[int, typer.Option("--batch-size")] = 8192,
	max_plan_edges: Annotated[int | None, typer.Option("--max-plan-edges")] = None,
	delta_max_steps: Annotated[int, typer.Option("--delta-max-steps")] = 32,
	history_bytes: Annotated[int, typer.Option("--history-bytes", help="Per-worker historical state budget")] = 256
	* 1024**2,
	delta_anchor_bytes: Annotated[
		int | None, typer.Option("--delta-anchor-bytes", help="Defaults to 75% of retained worker capacity")
	] = None,
	max_pending: Annotated[int, typer.Option("--max-pending")] = 32,
	max_rpc_bytes: Annotated[int, typer.Option("--max-rpc-bytes")] = 64 * 1024**2,
	max_tickets: Annotated[int, typer.Option("--max-tickets")] = 1024,
	ticket_retention: Annotated[float, typer.Option("--ticket-retention")] = 300,
	timeout: Annotated[float, typer.Option("--timeout")] = 30,
	retries: Annotated[int, typer.Option("--retries")] = 2,
	serial: Annotated[
		bool, typer.Option("--serial", help="Disable analysis lookahead for matched comparisons")
	] = False,
	features: Annotated[
		Path | None,
		typer.Option("--features", help="JSON object of boolean RuntimeFeatures; defaults apply to omitted fields"),
	] = None,
	profile_report: Annotated[Path | None, typer.Option("--profile-report")] = None,
	pipeline: Annotated[bool | None, typer.Option("--pipeline/--no-pipeline")] = None,
	coalescing: Annotated[bool | None, typer.Option("--coalescing/--no-coalescing")] = None,
	peer_fetch: Annotated[bool | None, typer.Option("--peer-fetch/--no-peer-fetch")] = None,
	gather_inputs: Annotated[bool | None, typer.Option("--gather-inputs/--no-gather-inputs")] = None,
	preemption: Annotated[bool | None, typer.Option("--preemption/--no-preemption")] = None,
	recovery: Annotated[bool | None, typer.Option("--recovery/--no-recovery")] = None,
	profiling: Annotated[bool | None, typer.Option("--profiling/--no-profiling")] = None,
	native_analysis: Annotated[bool | None, typer.Option("--native-analysis/--no-native-analysis")] = None,
	incremental_aggregates: Annotated[
		bool | None, typer.Option("--incremental-aggregates/--no-incremental-aggregates")
	] = None,
	bounded_staleness: Annotated[bool | None, typer.Option("--bounded-staleness/--no-bounded-staleness")] = None,
	worker_feature_cache: Annotated[
		bool | None, typer.Option("--worker-feature-cache/--no-worker-feature-cache")
	] = None,
	alltoall: Annotated[bool | None, typer.Option("--alltoall/--no-alltoall")] = None,
	elastic_groups: Annotated[bool | None, typer.Option("--elastic-groups/--no-elastic-groups")] = None,
	distributed_storage: Annotated[bool | None, typer.Option("--distributed-storage/--no-distributed-storage")] = None,
	concurrent_compute: Annotated[bool | None, typer.Option("--concurrent-compute/--no-concurrent-compute")] = None,
	retain_aggregates: Annotated[bool | None, typer.Option("--retain-aggregates/--no-retain-aggregates")] = None,
	aggregate_only: Annotated[bool | None, typer.Option("--aggregate-only/--no-aggregate-only")] = None,
	row_refresh: Annotated[bool | None, typer.Option("--row-refresh/--no-row-refresh")] = None,
	operator_incremental: Annotated[
		bool | None, typer.Option("--operator-incremental/--no-operator-incremental")
	] = None,
	pma_storage: Annotated[bool | None, typer.Option("--pma-storage/--no-pma-storage")] = None,
	mapped_host_reads: Annotated[bool | None, typer.Option("--mapped-host-reads/--no-mapped-host-reads")] = None,
):
	args = Options(
		**{
			**locals(),
			"backend": backend.value,
			"model": model.value if model is not None else None,
			"method": method.value if method is not None else None,
		}
	)
	if args.env is not None:
		config = json.loads(args.env.read_text())
		if not isinstance(config, dict) or any(
			not isinstance(value, (str, int, float, bool)) for value in config.values()
		):
			raise typer.BadParameter("Environment configuration must map names to scalar values")
		for key, value in config.items():
			os.environ.setdefault(key, str(value))
	try:
		placement = Placement.resolve(
			rank=args.rank, world_size=args.world_size, local_rank=args.local_rank, init_method=args.init_method
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	args.threads = (
		args.threads if args.threads is not None else int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
	)
	args.listen = (
		args.listen or f"{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_RPC_PORT', '29501')}"
	)
	_run(args, placement)


def _run(args: Options, placement: Placement):
	assert args.threads is not None
	switches = (
		RuntimeFeatures.from_dict(json.loads(args.features.read_text()))
		if args.features
		else RuntimeFeatures(pipeline=not args.serial)
	)
	switches = replace(
		switches,
		**{
			field.name: getattr(args, field.name)
			for field in fields(RuntimeFeatures)
			if getattr(args, field.name) is not None
		},
	)
	if args.serial and switches.pipeline:
		raise typer.BadParameter("--serial conflicts with pipeline=true in --features")
	if args.method is not None:
		incremental = args.method == NRT_INC and not switches.incremental_aggregates
		if args.operator_incremental is not None and args.operator_incremental != incremental:
			raise typer.BadParameter("--method conflicts with --operator-incremental")
		switches = replace(switches, operator_incremental=incremental)
	policy = neighborhood_policy(
		args.method or method_from_switches(switches.as_dict()), fanout=args.fanout, seed=args.sampling_seed
	)
	if (
		min(
			args.threads,
			args.native_threads,
			args.rpc_threads,
			args.batch_size,
			args.worker_bytes,
			args.max_pending,
			args.max_rpc_bytes,
			args.max_tickets,
			args.delta_max_steps,
		)
		< 1
		or min(args.storage_bytes, args.retries, args.ticket_retention, args.history_bytes) < 0
		or args.storage_node_capacity_bytes < 1
		or args.storage_node_reserve_bytes < 0
		or args.storage_node_reserve_bytes >= args.storage_node_capacity_bytes
		or args.storage_node_names is not None
		and len(args.storage_node_names) != args.storage_partitions
		or args.storage_endpoints is not None
		and len(args.storage_endpoints) != args.storage_partitions
		or args.delta_anchor_bytes is not None
		and args.delta_anchor_bytes < 0
	):
		raise typer.BadParameter("Counts must be positive; storage, retries, and retention must be nonnegative")
	if args.timeout <= 0 or args.max_plan_edges is not None and args.max_plan_edges < 0:
		raise typer.BadParameter("Invalid transport timeout or edge budget")
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	if switches.profiling and placement.rank in (None, 0) and args.profile_report is None:
		raise typer.BadParameter("--profile-report is required when profiling is enabled")
	settings = WorkerSettings(
		switches,
		args.threads,
		args.backend,
		args.timeout,
		history_bytes=args.history_bytes,
		refresh_steps=args.delta_max_steps,
		checkpoint=str(args.checkpoint) if args.checkpoint else None,
		rendezvous_prefix=args.rendezvous_prefix,
	)
	if placement.rank is not None and placement.rank != 0:
		assert placement.world_size is not None
		device = placement.worker_device(args.workers, args.device)
		run_worker(
			None,
			RemoteEndpoint(Owner(placement.rank, 1), torch.device(device), args.worker_bytes),
			placement.world_size,
			placement.init_method,
			settings,
		)
		return
	if args.data is None:
		raise typer.BadParameter("--data is required on the storage rank")
	devices = resolve_devices([args.device] if args.device else args.workers) if placement.rank is None else ()
	torch.set_num_threads(args.threads)
	graph = load_graph_data(Path(args.data), add_inverse_edges=args.add_inverse_edges)
	features = graph.ndata.pop("feat")
	graph.create_formats_()
	classes = args.classes
	if classes is None and args.checkpoint is None:
		labels = graph.ndata.get("label")
		if labels is not None:
			classes = int(labels.max()) + 1
	try:
		spec = ModelSpec.from_options(
			features.shape[1],
			checkpoint=str(args.checkpoint) if args.checkpoint else None,
			name=args.model,
			hidden=args.hidden,
			classes=classes,
			layers=args.layers,
			heads=args.heads,
			relations=args.relations,
			pseudo_dim=args.pseudo_dim,
			kernels=args.kernels,
			seed=args.seed,
		)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error

	spec.build()  # Validate checkpoint configuration before starting workers.
	stop = Event()
	with ExitStack() as stack:
		runtime = stack.enter_context(
			ManagedRuntime(
				graph,
				features,
				spec,
				devices,
				placement=placement,
				rendezvous_prefix=args.rendezvous_prefix,
				worker_threads=args.threads,
				history_bytes=args.history_bytes,
				backend=args.backend,
				worker_bytes=args.worker_bytes,
				storage_bytes=args.storage_bytes,
				storage_overlay_bytes=args.storage_overlay_bytes,
				storage_response_bytes=args.storage_response_bytes,
				timeout=args.timeout,
				max_retries=args.retries,
				batch_size=args.batch_size,
				max_plan_edges=args.max_plan_edges,
				switches=switches,
				native_threads=args.native_threads,
				storage_partitions=args.storage_partitions,
				storage_node_names=tuple(args.storage_node_names) if args.storage_node_names else None,
				storage_node_capacity_bytes=args.storage_node_capacity_bytes,
				storage_node_reserve_bytes=args.storage_node_reserve_bytes,
				storage_endpoints=tuple(args.storage_endpoints) if args.storage_endpoints else None,
				storage_authkey=args.storage_authkey,
				storage_directory=str(args.topology_root) if args.topology_root else None,
				max_stale_versions=args.max_stale_versions,
				idle_flush_seconds=args.idle_flush_seconds,
				delta_max_steps=args.delta_max_steps,
				delta_anchor_bytes=args.delta_anchor_bytes,
				neighborhood_policy=policy,
			)
		)
		graph = runtime.service.graph
		if switches.profiling:
			profiler = stack.enter_context(RuntimeProfiler(runtime, network_interfaces=args.network_interfaces))
			profiler.start()
			stack.callback(_export_profile, profiler, args.profile_report, switches)
		frontend = OwnedServicer(
			runtime.service,
			runtime.service.remote_plane.storage.owner,
			max_pending=args.max_pending,
			max_bytes=args.max_rpc_bytes,
			max_tickets=args.max_tickets,
			ticket_retention=args.ticket_retention,
		)
		stack.callback(frontend.close)
		executor = stack.enter_context(ThreadPoolExecutor(max_workers=args.rpc_threads))
		from lib.rpc.timing import ServerTiming

		server = grpc.server(
			executor,
			options=message_options(args.max_rpc_bytes),
			interceptors=[ServerTiming()] if args.rpc_timing else (),
		)
		stack.callback(lambda: server.stop(5).wait())
		add_OwnedInferenceServicer_to_server(frontend, server)
		port = server.add_insecure_port(args.listen)
		if not port:
			raise RuntimeError(f"Could not bind inference RPC listener {args.listen}")
		for signum in (signal.SIGINT, signal.SIGTERM):
			signal.signal(signum, lambda _signum, _frame: stop.set())
		server.start()
		if args.ready_file is not None:
			write_ready(args.ready_file, args.listen, port, frontend.tickets.session)
		logging.info("Runtime feature switches: %s", switches.as_dict())
		logging.info(
			"Inference RPC listening at %s (port %s), nodes=%s, session=%s",
			args.listen,
			port,
			graph.num_nodes(),
			frontend.tickets.session,
		)
		stop.wait()
		logging.info("Draining admitted inference requests before stopping managed workers")


if __name__ == "__main__":
	app()
