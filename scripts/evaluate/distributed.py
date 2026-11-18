"""Run matched inference experiments on ranks placed by torchrun or CLI topology."""

import hashlib
import json
import logging
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import dgl
import grpc
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.methods import (
	METHODS,
	NRT_INC,
	neighborhood_policy,
	validate_method,
)
from lib.graph.ownership import Owner
from lib.models import MODEL_NAMES
from lib.models.spec import ModelSpec
from lib.profiling.accuracy import load_test_labels, query_error, targeted_sample, verification_sample
from lib.profiling.controller import RuntimeProfiler
from lib.profiling.experiment import (
	inference_identity,
	latency_summary,
	paired_comparison,
	sample_ids,
	source_fingerprint,
)
from lib.profiling.metrics import measurements
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceStub, add_OwnedInferenceServicer_to_server
from lib.rpc.limits import message_options
from lib.rpc.service import OwnedServicer, _from_wire, _to_wire
from lib.rpc.timing import RpcTimings, ServerTiming
from lib.runtime.config import RuntimeFeatures
from lib.runtime.launch import Placement, WorkerSettings, run_worker
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import RemoteEndpoint
from lib.storage.capacity import GIB

NodeSelectionChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"NodeSelectionChoice", {str(value).replace("-", "_").upper(): value for value in ["neighbors", "prefix"]}
)
ModelsChoice = StrEnum("ModelsChoice", {str(value).replace("-", "_").upper(): value for value in MODEL_NAMES})  # type: ignore[misc]  # Mypy requires literal Enum members.
MethodsChoice = StrEnum("MethodsChoice", {str(value).replace("-", "_").upper(): value for value in METHODS})  # type: ignore[misc]  # Mypy requires literal Enum members.
WorkloadChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"WorkloadChoice", {str(value).replace("-", "_").upper(): value for value in ["mixed", "edges"]}
)
BackendChoice = StrEnum("BackendChoice", {str(value).replace("-", "_").upper(): value for value in ["auto", "gloo"]})  # type: ignore[misc]  # Mypy requires literal Enum members.
TopologyChoice = StrEnum("TopologyChoice", {str(value).replace("-", "_").upper(): value for value in ["dgl", "mapped"]})  # type: ignore[misc]  # Mypy requires literal Enum members.
IncrementalEngineChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"IncrementalEngineChoice",
	{str(value).replace("-", "_").upper(): value for value in ["auto", "operators", "aggregates"]},
)
InitializationChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"InitializationChoice", {str(value).replace("-", "_").upper(): value for value in ["queries", "all"]}
)
DispatchChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DispatchChoice", {str(value).replace("-", "_").upper(): value for value in ["round_robin", "locality_load"]}
)
DeferredPolicyChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DeferredPolicyChoice",
	{str(value).replace("-", "_").upper(): value for value in ["disabled", "version_window", "fixed_time", "hotspot"]},
)


_DEFAULT_NODE_SELECTION = NodeSelectionChoice("neighbors")
_DEFAULT_WORKLOAD = WorkloadChoice("mixed")
_DEFAULT_BACKEND = BackendChoice("auto")
_DEFAULT_TOPOLOGY = TopologyChoice("dgl")
_DEFAULT_INCREMENTAL_ENGINE = IncrementalEngineChoice("auto")
_DEFAULT_INITIALIZATION = InitializationChoice("all")
_DEFAULT_DISPATCH = DispatchChoice("round_robin")
_DEFAULT_DEFERRED_POLICY = DeferredPolicyChoice("disabled")


@dataclass
class Options:
	output: Path
	data: Path | None
	add_inverse_edges: bool
	node_limit: int | None
	node_selection: str
	nodes: int
	edges: int
	features: int
	hidden: int
	classes: int
	layers: int
	heads: int
	relations: int
	pseudo_dim: int
	models: list[str]
	methods: list[str]
	repetitions: int
	repetition_offset: int
	phase_file: Path | None
	fanout: int
	warmup: int
	updates: int
	changes: int
	change_fraction: float | None
	workload: str
	checkpoint: Path | None
	atol: float
	rtol: float
	queries: int
	verification_rows: int
	verification_batch_size: int
	seed: int
	threads: int
	device: str | None
	rank: int | None
	world_size: int | None
	local_rank: int | None
	init_method: str | None
	backend: str
	timeout: float
	worker_bytes: int
	storage_bytes: int
	storage_overlay_bytes: int
	storage_response_bytes: int
	history_bytes: int
	anchor_bytes: int | None
	refresh_steps: int
	batch_size: int
	max_plan_edges: int | None
	coalescing: bool
	pipeline: bool
	gather_inputs: bool
	concurrent_compute: bool
	topology: str
	topology_root: Path | None
	topology_base: Path | None
	storage_partitions: int
	storage_node_names: list[str] | None
	storage_node_capacity_bytes: int
	storage_node_reserve_bytes: int
	storage_endpoints: list[str] | None
	storage_authkey: str
	network_interfaces: list[str] | None
	incremental_engine: str
	initialization: str
	initialization_batch_size: int
	dispatch_policy: str
	dispatch_locality_weight: float
	dispatch_load_weight: float
	worker_feature_cache: bool
	worker_feature_cache_bytes: int
	deferred_mode: str
	max_stale_versions: int
	deferred_window_seconds: float | None
	hotspot_degree_threshold: int | None
	hotspot_max_changes: int | None
	source_fingerprint: str = ""
	labels: Path | None = None
	test_split: Path | None = None


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	output: Annotated[
		Path, typer.Option("--output", help="New results directory; existing files are never overwritten")
	],
	data: Annotated[Path | None, typer.Option("--data")] = None,
	add_inverse_edges: Annotated[
		bool, typer.Option("--add-inverse-edges", help="Include reverse edges, including self loops")
	] = False,
	node_limit: Annotated[int | None, typer.Option("--node-limit")] = None,
	node_selection: Annotated[NodeSelectionChoice, typer.Option("--node-selection")] = _DEFAULT_NODE_SELECTION,
	nodes: Annotated[int, typer.Option("--nodes")] = 256,
	edges: Annotated[int, typer.Option("--edges")] = 2048,
	features: Annotated[int, typer.Option("--features")] = 16,
	hidden: Annotated[int, typer.Option("--hidden")] = 16,
	classes: Annotated[int, typer.Option("--classes")] = 8,
	layers: Annotated[int, typer.Option("--layers")] = 2,
	heads: Annotated[int, typer.Option("--heads")] = 2,
	relations: Annotated[int, typer.Option("--relations")] = 4,
	pseudo_dim: Annotated[int, typer.Option("--pseudo-dim")] = 2,
	models: Annotated[list[ModelsChoice] | None, typer.Option("--models")] = None,
	methods: Annotated[
		list[MethodsChoice] | None, typer.Option("--methods", help="Inference algorithm to execute.")
	] = None,
	repetitions: Annotated[int, typer.Option("--repetitions")] = 5,
	repetition_offset: Annotated[int, typer.Option("--repetition-offset", min=0)] = 0,
	phase_file: Annotated[Path | None, typer.Option("--phase-file")] = None,
	fanout: Annotated[int, typer.Option("--fanout", min=1, help="Neighbors per node at every layer of RTEC-NS")] = 5,
	warmup: Annotated[int, typer.Option("--warmup")] = 3,
	updates: Annotated[int, typer.Option("--updates")] = 6,
	changes: Annotated[int, typer.Option("--changes")] = 4,
	change_fraction: Annotated[
		float | None, typer.Option("--change-fraction", help="Edge changes per batch as a fraction of initial edges")
	] = None,
	workload: Annotated[WorkloadChoice, typer.Option("--workload")] = _DEFAULT_WORKLOAD,
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	labels: Annotated[Path | None, typer.Option("--labels", exists=True)] = None,
	test_split: Annotated[Path | None, typer.Option("--test-split", exists=True)] = None,
	atol: Annotated[float, typer.Option("--atol")] = 5e-05,
	rtol: Annotated[float, typer.Option("--rtol")] = 5e-05,
	queries: Annotated[int, typer.Option("--queries")] = 64,
	verification_rows: Annotated[
		int, typer.Option("--verification-rows", help="Fixed oracle sample; 0 verifies every node")
	] = 4096,
	verification_batch_size: Annotated[
		int, typer.Option("--verification-batch-size", help="Maximum rows per verification RPC")
	] = 512,
	seed: Annotated[int, typer.Option("--seed")] = 71,
	threads: Annotated[int, typer.Option("--threads")] = int(
		os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())
	),
	device: Annotated[
		str | None, typer.Option("--device", help="Worker device within this process visible devices")
	] = None,
	rank: Annotated[int | None, typer.Option("--rank")] = None,
	world_size: Annotated[int | None, typer.Option("--world-size")] = None,
	local_rank: Annotated[int | None, typer.Option("--local-rank")] = None,
	init_method: Annotated[str | None, typer.Option("--init-method")] = None,
	backend: Annotated[BackendChoice, typer.Option("--backend")] = _DEFAULT_BACKEND,
	timeout: Annotated[float, typer.Option("--timeout")] = 120,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 512 * 1024**2,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 1024**3,
	storage_overlay_bytes: Annotated[
		int,
		typer.Option(
			"--storage-overlay-bytes", min=0, help="Changed adjacency bytes per direction and storage partition"
		),
	] = 256 * 1024**2,
	storage_response_bytes: Annotated[
		int, typer.Option("--storage-response-bytes", min=0, help="Maximum selected adjacency bytes per read operation")
	] = 256 * 1024**2,
	history_bytes: Annotated[int, typer.Option("--history-bytes")] = 256 * 1024**2,
	anchor_bytes: Annotated[
		int | None,
		typer.Option("--anchor-bytes", help="Pinned old-state limit; defaults to 75% of retained worker capacity"),
	] = None,
	refresh_steps: Annotated[int, typer.Option("--refresh-steps")] = 32,
	batch_size: Annotated[int, typer.Option("--batch-size")] = 256,
	max_plan_edges: Annotated[
		int | None,
		typer.Option(
			"--max-plan-edges", min=0, help="Dependency and sparse-route edge budget; defaults to the runtime limits"
		),
	] = None,
	coalescing: Annotated[bool, typer.Option("--coalescing/--no-coalescing")] = True,
	pipeline: Annotated[bool, typer.Option("--pipeline/--no-pipeline")] = True,
	gather_inputs: Annotated[bool, typer.Option("--gather-inputs/--no-gather-inputs")] = True,
	concurrent_compute: Annotated[bool, typer.Option("--concurrent-compute/--no-concurrent-compute")] = False,
	topology: Annotated[TopologyChoice, typer.Option("--topology")] = _DEFAULT_TOPOLOGY,
	topology_root: Annotated[
		Path | None, typer.Option("--topology-root", help="Working directory for mapped adjacency files")
	] = None,
	topology_base: Annotated[
		Path | None,
		typer.Option(
			"--topology-base",
			help="Existing exported mapped adjacency directory; required with remote storage endpoints",
		),
	] = None,
	storage_partitions: Annotated[int, typer.Option("--storage-partitions")] = 1,
	storage_node_names: Annotated[
		list[str] | None,
		typer.Option("--storage-node", help="Storage node label; repeat once per storage partition"),
	] = None,
	storage_node_capacity_bytes: Annotated[
		int, typer.Option("--storage-node-capacity-bytes", help="Physical capacity reported for each storage node")
	] = 512 * GIB,
	storage_node_reserve_bytes: Annotated[
		int, typer.Option("--storage-node-reserve-bytes", help="Memory reserved for non-NeutronRT workloads")
	] = 64 * GIB,
	storage_endpoints: Annotated[
		list[str] | None,
		typer.Option("--storage-endpoint", help="Remote storage HOST:PORT; repeat once per storage partition"),
	] = None,
	storage_authkey: Annotated[str, typer.Option("--storage-authkey")] = "neutronrt",
	network_interfaces: Annotated[
		list[str] | None,
		typer.Option(
			"--network-interface", help="Interface to sample for OS network counters; repeat to select several"
		),
	] = None,
	incremental_engine: Annotated[
		IncrementalEngineChoice, typer.Option("--incremental-engine")
	] = _DEFAULT_INCREMENTAL_ENGINE,
	initialization: Annotated[InitializationChoice, typer.Option("--initialization")] = _DEFAULT_INITIALIZATION,
	initialization_batch_size: Annotated[
		int,
		typer.Option(
			"--initialization-batch-size", min=1, help="Rows per bulk initialization request, before task splitting"
		),
	] = 65536,
	dispatch_policy: Annotated[DispatchChoice, typer.Option("--dispatch-policy")] = _DEFAULT_DISPATCH,
	dispatch_locality_weight: Annotated[float, typer.Option("--dispatch-locality-weight", min=0)] = 1.0,
	dispatch_load_weight: Annotated[float, typer.Option("--dispatch-load-weight", min=0)] = 1.0,
	worker_feature_cache: Annotated[bool, typer.Option("--worker-feature-cache/--no-worker-feature-cache")] = False,
	worker_feature_cache_bytes: Annotated[int, typer.Option("--worker-feature-cache-bytes", min=0)] = 0,
	deferred_mode: Annotated[DeferredPolicyChoice, typer.Option("--deferred-mode")] = _DEFAULT_DEFERRED_POLICY,
	max_stale_versions: Annotated[int, typer.Option("--max-stale-versions", min=0)] = 2,
	deferred_window_seconds: Annotated[float | None, typer.Option("--deferred-window-seconds", min=0)] = None,
	hotspot_degree_threshold: Annotated[int | None, typer.Option("--hotspot-degree-threshold", min=0)] = None,
	hotspot_max_changes: Annotated[int | None, typer.Option("--hotspot-max-changes", min=1)] = None,
):
	args = Options(
		**{
			**locals(),
			"node_selection": node_selection.value,
			"models": [item.value for item in models] if models is not None else list(MODEL_NAMES),
			"methods": [item.value for item in methods] if methods is not None else list(METHODS),
			"workload": workload.value,
			"backend": backend.value,
			"topology": topology.value,
			"storage_node_names": storage_node_names,
			"storage_endpoints": storage_endpoints,
			"incremental_engine": incremental_engine.value,
			"initialization": initialization.value,
			"dispatch_policy": dispatch_policy.value,
			"deferred_mode": deferred_mode.value,
		}
	)
	if args.topology == "mapped" and any(name in ("pinsage", "rgcn", "rgat", "monet") for name in args.models):
		raise typer.BadParameter("Mapped topology does not yet store model-specific edge attributes")
	if args.storage_node_names is not None and len(args.storage_node_names) != args.storage_partitions:
		raise typer.BadParameter("Pass one --storage-node label per storage partition")
	if args.storage_endpoints is not None and len(args.storage_endpoints) != args.storage_partitions:
		raise typer.BadParameter("Pass one --storage-endpoint per storage partition")
	if args.storage_endpoints is not None and args.topology_base is None:
		raise typer.BadParameter(
			"Remote mapped storage requires --topology-base pointing to an existing BaseTopology.export directory"
		)
	if args.dispatch_locality_weight == 0 and args.dispatch_load_weight == 0:
		raise typer.BadParameter("At least one dispatch policy weight must be positive")
	if args.worker_feature_cache and args.worker_feature_cache_bytes < 1:
		raise typer.BadParameter("--worker-feature-cache requires a positive --worker-feature-cache-bytes budget")
	if not args.worker_feature_cache and args.worker_feature_cache_bytes:
		raise typer.BadParameter("--worker-feature-cache-bytes requires --worker-feature-cache")
	if args.worker_feature_cache:
		raise typer.BadParameter(
			"--worker-feature-cache is currently unsupported: the wire protocol has no miss-only cache path"
		)
	if args.deferred_mode == "disabled":
		if (
			args.max_stale_versions != 2
			or args.deferred_window_seconds is not None
			or args.hotspot_degree_threshold is not None
			or args.hotspot_max_changes is not None
		):
			raise typer.BadParameter(
				"Deferred policy options require --deferred-mode version_window, fixed_time, or hotspot"
			)
	elif args.deferred_mode == "version_window":
		if (
			args.deferred_window_seconds is not None
			or args.hotspot_degree_threshold is not None
			or args.hotspot_max_changes is not None
		):
			raise typer.BadParameter("Version-window mode accepts only --max-stale-versions")
	elif args.deferred_mode == "fixed_time":
		if args.deferred_window_seconds is None or args.deferred_window_seconds <= 0:
			raise typer.BadParameter("Fixed-time mode requires a positive --deferred-window-seconds")
		if args.hotspot_degree_threshold is not None or args.hotspot_max_changes is not None:
			raise typer.BadParameter("Fixed-time mode does not accept hotspot options")
	else:
		if args.deferred_window_seconds is None or args.deferred_window_seconds <= 0:
			raise typer.BadParameter("Hotspot mode requires a positive --deferred-window-seconds")
		if args.hotspot_degree_threshold is None or args.hotspot_max_changes is None:
			raise typer.BadParameter("Hotspot mode requires degree threshold and maximum changes")
	if (
		args.storage_node_capacity_bytes < 1
		or args.storage_node_reserve_bytes < 0
		or args.storage_node_reserve_bytes >= args.storage_node_capacity_bytes
	):
		raise typer.BadParameter("Storage capacity must exceed its non-NeutronRT reserve")
	if args.incremental_engine == "aggregates" and any(name != "sage" for name in args.models):
		raise typer.BadParameter("Projected aggregates require mean GraphSAGE; use auto for mixed models")
	if len(set(args.methods)) != len(args.methods):
		raise typer.BadParameter("Methods must be distinct")
	if (
		min(
			args.nodes,
			args.edges,
			args.features,
			args.hidden,
			args.classes,
			args.layers,
			args.repetitions,
			args.updates,
			args.changes,
			args.threads,
			args.batch_size,
			args.storage_partitions,
			args.fanout,
		)
		< 1
		or args.warmup < 0
		or args.queries < 0
		or args.verification_rows < 0
		or args.verification_batch_size < 1
		or args.node_limit is not None
		and args.node_limit < 1
		or args.change_fraction is not None
		and not 0 < args.change_fraction <= 1
		or min(args.atol, args.rtol) < 0
	):
		raise typer.BadParameter("Graph dimensions and trial limits must be positive")
	_run(args)


def load_graph(args):
	generator = torch.Generator().manual_seed(args.seed)
	if args.data:
		graph = load_graph_data(Path(args.data), add_inverse_edges=args.add_inverse_edges)
		if args.node_limit is not None:
			limit = min(args.node_limit, graph.num_nodes())
			selected = torch.arange(limit)
			if args.node_selection == "neighbors":
				selected = torch.randint(graph.num_nodes(), (1,), generator=generator)
				frontier = selected
				while selected.numel() < limit:
					neighbors = torch.cat([graph.in_edges(frontier)[0], graph.out_edges(frontier)[1]]).unique()
					candidates = neighbors[~torch.isin(neighbors, selected)]
					if not candidates.numel():
						candidates = torch.arange(graph.num_nodes())
						candidates = candidates[~torch.isin(candidates, selected)]
					frontier = candidates[
						torch.randperm(candidates.numel(), generator=generator)[: limit - selected.numel()]
					]
					selected = torch.cat([selected, frontier])
			graph = dgl.node_subgraph(graph, selected.sort().values)
		features = graph.ndata.pop("feat").float()
	else:
		u, v = torch.randint(args.nodes, (2, args.edges), generator=generator)
		graph = dgl.graph((u, v), num_nodes=args.nodes)
		features = torch.randn(args.nodes, args.features, generator=generator) * 0.1
	return graph, features


def attributes(count, spec, generator):
	values = torch.randn(count, spec.pseudo_dim + 2, generator=generator)
	values[:, 0] = torch.randint(spec.relations if spec.name in ("rgcn", "rgat") else 1, (count,), generator=generator)
	values[:, 1] = torch.rand(count, generator=generator) + 0.5
	return values


@torch.inference_mode()
def verify(
	stub,
	model,
	graph,
	features,
	*,
	atol=5e-5,
	rtol=5e-5,
	exact=True,
	timeout=180,
	nodes=None,
	batch_size=512,
	labels=None,
	test_mask=None,
):
	if batch_size < 1:
		raise ValueError("Verification batch size must be positive")
	nodes = torch.arange(graph.num_nodes()) if nodes is None else nodes
	if not nodes.numel():
		raise ValueError("Verification requires at least one row")
	# Independent full-graph oracle, outside the measured execution window.
	expected = features
	for layer in range(model.num_layers):
		expected = model(layer, graph, expected)
	squared_error, elements, max_error, disagreement = 0.0, 0, 0.0, 0
	version = None
	labelled, correct, fresh_correct = 0, 0, 0
	# Leave room for protobuf framing in the 64 MiB RPC response budget.
	rows_per_call = min(batch_size, (60 * 1024**2) // (4 * expected.shape[1]))
	if rows_per_call < 1:
		raise ValueError("One output row exceeds the verification response budget")
	for chunk in nodes.split(rows_per_call):
		response = stub.Query(wire.QueryRequest(nodes=_to_wire(chunk)), timeout=timeout)
		if version is not None and response.version != version:
			raise RuntimeError("Graph version changed during verification")
		version = response.version
		actual = _from_wire(response.values, 64 * 1024**2)
		reference = expected[chunk]
		if exact:
			torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
		errors = query_error(actual, reference)
		squared_error += errors["mean_squared_error"] * actual.numel()
		elements += actual.numel()
		max_error = max(max_error, errors["max_abs_error"])
		disagreement += int((actual.argmax(1) != reference.argmax(1)).sum())
		if labels is not None:
			selected = test_mask[chunk] & (labels[chunk] >= 0) & (labels[chunk] < actual.shape[1])
			labelled += int(selected.sum())
			correct += int((actual.argmax(1)[selected] == labels[chunk][selected]).sum())
			fresh_correct += int((reference.argmax(1)[selected] == labels[chunk][selected]).sum())
	mse = squared_error / elements
	if exact and mse >= 1e-4:
		raise AssertionError(f"Embedding MSE {mse} exceeds the embedding MSE limit (1e-4)")
	if exact and disagreement:
		raise AssertionError(f"{disagreement} predictions disagree with full inference")
	return {
		"version": version,
		"rows": nodes.numel(),
		"graph_nodes": graph.num_nodes(),
		"oracle_scope": "all rows" if nodes.numel() == graph.num_nodes() else "fixed sampled rows",
		"max_error": max_error,
		"mean_squared_error": mse,
		"absolute_tolerance": atol,
		"relative_tolerance": rtol,
		"prediction_disagreements": disagreement,
		"exact_equivalence_required": exact,
		"labelled_test_rows": labelled,
		"sample_test_accuracy": correct / labelled if labelled else None,
		"fresh_sample_test_accuracy": fresh_correct / labelled if labelled else None,
		"accuracy_loss_percentage_points": 100 * (fresh_correct - correct) / labelled if labelled else None,
	}


def make_update(graph, features, spec, index, changes, generator, request_id, *, workload, pending=None):
	operation = index % 3 if workload == "mixed" else 1 + index % 2
	count = min(changes, graph.num_nodes()) if operation == 0 else changes
	kwargs = {}
	if operation == 0:
		kind = wire.Update.FEATURES
		first = sample_ids(graph.num_nodes(), count, generator)
		second = features[first].clone()
		if pending is not None:
			for row, node in enumerate(first.tolist()):
				if node in pending:
					second[row] = pending[node]
		second += torch.randn(count, features.shape[1], generator=generator) * 0.05
		if pending is not None:
			pending.update(zip(first.tolist(), second.clone().unbind(), strict=True))
	elif operation == 1:
		kind = wire.Update.INSERT
		first = torch.randint(graph.num_nodes(), (count,), generator=generator)
		second = torch.randint(graph.num_nodes(), (count,), generator=generator)
		if "nrt" in graph.edata:
			kwargs["attributes"] = _to_wire(attributes(count, spec, generator))
	else:
		kind = wire.Update.DELETE
		eids = sample_ids(graph.num_edges(), min(count, graph.num_edges()), generator)
		first, second = graph.find_edges(eids)
	return wire.Update(kind=kind, first=_to_wire(first), second=_to_wire(second), request_id=request_id, **kwargs)


def mirror_edges(graph, request):
	if request.kind == wire.Update.FEATURES:
		return
	first, second = (_from_wire(value, 64 * 1024**2) for value in (request.first, request.second))
	if request.kind == wire.Update.INSERT:
		data = {"nrt": _from_wire(request.attributes, 64 * 1024**2)} if request.HasField("attributes") else None
		graph.add_edges(first, second, data=data)
	else:
		graph.remove_edges(graph.edge_ids(first, second, return_uv=True)[2].unique())


def edge_change_record(graph, request, *, mirror: bool) -> dict:
	"""Record requested pairs and effective graph-edge changes separately."""
	requested = int(request.first.shape[0])
	before = graph.num_edges()
	if mirror:
		mirror_edges(graph, request)
	after = graph.num_edges()
	return {
		"kind": int(request.kind),
		"requested_changes": requested,
		"edge_count_before": before,
		"edge_count_after": after,
		"effective_edge_delta": after - before if request.kind != wire.Update.FEATURES else 0,
		"effective_edge_changes": abs(after - before) if request.kind != wire.Update.FEATURES else 0,
	}


def execute_update(stub, request, timeout, *, duplicate=False):
	start = perf_counter()
	if request.kind == wire.Update.INSERT:
		ticket = stub.Async(request, timeout=timeout)
		admission = perf_counter() - start
		completion = stub.Wait(wire.WaitRequest(ticket=ticket), timeout=timeout)
	else:
		completion = stub.Sync(request, timeout=timeout)
		admission = None
	elapsed = perf_counter() - start
	if duplicate:
		repeated = stub.Sync(request, timeout=timeout)
		assert repeated.version == completion.version
		if request.HasField("attributes"):
			changed = wire.Update()
			changed.CopyFrom(request)
			data = _from_wire(changed.attributes, 64 * 1024**2).clone()
			data[:, 1] += 0.1
			changed.attributes.CopyFrom(_to_wire(data))
			try:
				stub.Sync(changed, timeout=timeout)
			except grpc.RpcError as error:
				assert error.code() == grpc.StatusCode.INVALID_ARGUMENT
			else:
				raise AssertionError("Changed edge attributes reused a request ID")
	stub.Release(completion.ticket, timeout=timeout)
	return {
		"kind": request.kind,
		"version": completion.version,
		"changes": completion.input_changes,
		"client_seconds": elapsed,
		"admission_seconds": admission,
	}


def trial(args, placement, base_graph, base_features, name, method, repetition):
	incremental = validate_method(method) == NRT_INC
	dispatch_mode = getattr(args, "dispatch_policy", "round_robin")
	deferred_mode = getattr(args, "deferred_mode", "disabled")
	worker_cache = getattr(args, "worker_feature_cache", False)
	worker_cache_bytes = getattr(args, "worker_feature_cache_bytes", 0)
	aggregates = incremental and (
		args.incremental_engine == "aggregates" or args.incremental_engine == "auto" and name == "sage"
	)
	switches = RuntimeFeatures(
		profiling=True,
		operator_incremental=incremental and not aggregates,
		retain_aggregates=aggregates,
		incremental_aggregates=aggregates,
		row_refresh=aggregates,
		distributed_storage=args.topology == "mapped",
		pipeline=args.pipeline,
		coalescing=args.coalescing,
		gather_inputs=args.gather_inputs,
		concurrent_compute=args.concurrent_compute,
		bounded_staleness=deferred_mode != "disabled",
		worker_feature_cache=worker_cache,
	)
	settings = WorkerSettings(
		switches,
		threads=args.threads,
		backend=args.backend,
		timeout=args.timeout,
		history_bytes=args.history_bytes,
		refresh_steps=args.refresh_steps,
		feature_cache_bytes=worker_cache_bytes,
		rendezvous_prefix=f"{name}/{method}/{repetition}",
	)
	if placement.rank:
		device = placement.worker_device(device=args.device)
		run_worker(
			None,
			RemoteEndpoint(Owner(placement.rank, 1), torch.device(device), args.worker_bytes),
			placement.world_size,
			placement.init_method,
			settings,
		)
		return None
	phase(args, method, repetition, "setup")
	# Give each trial its own lazy layout caches and mutable feature state.
	graph = base_graph.formats("coo").clone().formats(["coo", "csr", "csc"])
	features = base_features if args.workload == "edges" else base_features.clone()
	policy = neighborhood_policy(method, fanout=args.fanout, seed=args.seed + repetition)
	spec = ModelSpec(
		name,
		features.shape[1],
		args.hidden,
		args.classes,
		layers=args.layers,
		heads=args.heads,
		relations=args.relations,
		pseudo_dim=args.pseudo_dim,
		checkpoint=str(args.checkpoint) if args.checkpoint else None,
	)
	generator = torch.Generator().manual_seed(args.seed + repetition)
	if name in ("pinsage", "rgcn", "rgat", "monet"):
		graph.edata["nrt"] = attributes(graph.num_edges(), spec, generator)
	model = spec.build()
	changes = max(1, round(graph.num_edges() * args.change_fraction)) if args.change_fraction else args.changes
	input_identity = inference_identity(graph, features, spec, model=model)
	policy_identity = {
		"dispatch_policy": dispatch_mode,
		"dispatch_locality_weight": getattr(args, "dispatch_locality_weight", 1.0),
		"dispatch_load_weight": getattr(args, "dispatch_load_weight", 1.0),
		"worker_feature_cache": worker_cache,
		"worker_feature_cache_bytes": worker_cache_bytes,
		"deferred_mode": deferred_mode,
		"max_stale_versions": getattr(args, "max_stale_versions", 2),
		"deferred_window_seconds": getattr(args, "deferred_window_seconds", None),
		"hotspot_degree_threshold": getattr(args, "hotspot_degree_threshold", None),
		"hotspot_max_changes": getattr(args, "hotspot_max_changes", None),
	}
	identity = hashlib.sha256(
		(
			input_identity
			+ args.source_fingerprint
			+ json.dumps(policy_identity, sort_keys=True, separators=(",", ":"))
		).encode()
	).hexdigest()
	digest = hashlib.sha256()
	with ExitStack() as stack:
		runtime = stack.enter_context(
			ManagedRuntime(
				graph,
				features,
				spec,
				placement=placement,
				switches=switches,
				worker_threads=args.threads,
				worker_bytes=args.worker_bytes,
				worker_feature_cache_bytes=worker_cache_bytes,
				dispatch_mode=dispatch_mode,
				dispatch_locality_weight=getattr(args, "dispatch_locality_weight", 1.0),
				dispatch_load_weight=getattr(args, "dispatch_load_weight", 1.0),
				deferred_mode="version_window" if deferred_mode == "disabled" else deferred_mode,
				max_stale_versions=getattr(args, "max_stale_versions", 2),
				deferred_window_seconds=getattr(args, "deferred_window_seconds", None),
				hotspot_degree_threshold=getattr(args, "hotspot_degree_threshold", None),
				hotspot_max_changes=getattr(args, "hotspot_max_changes", None),
				storage_bytes=args.storage_bytes,
				storage_overlay_bytes=args.storage_overlay_bytes,
				storage_response_bytes=args.storage_response_bytes,
				storage_node_names=tuple(args.storage_node_names) if args.storage_node_names else None,
				storage_node_capacity_bytes=args.storage_node_capacity_bytes,
				storage_node_reserve_bytes=args.storage_node_reserve_bytes,
				storage_endpoints=tuple(args.storage_endpoints) if args.storage_endpoints else None,
				storage_authkey=args.storage_authkey,
				history_bytes=args.history_bytes,
				delta_max_steps=args.refresh_steps,
				delta_anchor_bytes=args.anchor_bytes,
				backend=args.backend,
				timeout=args.timeout,
				batch_size=args.batch_size,
				max_plan_edges=args.max_plan_edges,
				rendezvous_prefix=settings.rendezvous_prefix,
				neighborhood_policy=policy,
				storage_partitions=args.storage_partitions,
				storage_directory=str(args.topology_root) if args.topology_root else None,
				storage_exported_base=str(args.topology_base) if args.topology_base else None,
			)
		)
		frontend = OwnedServicer(runtime.service, runtime.plane.storage.owner)
		stack.callback(frontend.close)
		executor = stack.enter_context(ThreadPoolExecutor(max_workers=4))
		server = grpc.server(executor, options=message_options(), interceptors=[ServerTiming()])
		add_OwnedInferenceServicer_to_server(frontend, server)
		port = server.add_insecure_port("127.0.0.1:0")
		server.start()
		stack.callback(lambda: server.stop(5).wait())
		channel = stack.enter_context(grpc.insecure_channel(f"127.0.0.1:{port}", options=message_options()))
		rpc_timings = RpcTimings()
		stub = OwnedInferenceStub(grpc.intercept_channel(channel, rpc_timings))
		if args.topology == "mapped":
			# An independent mutable DGL oracle mirrors actor updates outside RPC timing.
			graph = graph.clone()
		phase(args, method, repetition, "initialization")
		initialization_started = perf_counter()
		if args.initialization == "all":
			layers = range(1, spec.layers + 1) if policy.reuse_intermediates else (spec.layers,)
			for layer in layers:
				for start in range(0, graph.num_nodes(), args.initialization_batch_size):
					stop = min(start + args.initialization_batch_size, graph.num_nodes())
					runtime.service.ensure(torch.arange(start, stop), layer)
					if start == 0 or start // args.initialization_batch_size % 128 == 0 or stop == graph.num_nodes():
						logging.info(
							"Initialization: method=%s layer=%d rows=%d/%d", method, layer, stop, graph.num_nodes()
						)
		initialization_seconds = perf_counter() - initialization_started

		verification_generator = torch.Generator().manual_seed(args.seed + repetition)
		verification_nodes = verification_sample(
			graph,
			min(args.verification_rows or graph.num_nodes(), graph.num_nodes()),
			verification_generator,
		)

		labels, test_mask = load_test_labels(
			getattr(args, "labels", None), getattr(args, "test_split", None), graph.num_nodes()
		)

		def check():
			logging.info("Checking %d oracle rows for %s", verification_nodes.numel(), method)
			checked = verify(
				stub,
				model,
				graph,
				features,
				atol=args.atol,
				rtol=args.rtol,
				exact=not policy.approximate,
				timeout=args.timeout,
				nodes=verification_nodes,
				batch_size=args.verification_batch_size,
				labels=labels,
				test_mask=test_mask,
			)
			checked["selection"] = (
				"all rows"
				if verification_nodes.numel() == graph.num_nodes()
				else ("uniform rows and baseline indegree/outdegree extremes")
			)
			logging.info("Oracle check complete: method=%s max_error=%.6g", method, checked["max_error"])
			return checked

		phase(args, method, repetition, "initial_verification")
		checks = [check()]
		phase(args, method, repetition, "warmup")
		warmup_requests = []
		warmup_edge_changes = []
		for index in range(args.warmup):
			request = make_update(
				graph, features, spec, index, changes, generator, f"warm-{index}", workload=args.workload
			)
			digest.update(request.SerializeToString(deterministic=True))
			warmup_requests.append(request)
			before_edges = graph.num_edges()
			execute_update(stub, request, args.timeout, duplicate=True)
			if args.topology == "mapped":
				mirror_edges(graph, request)
			after_edges = graph.num_edges()
			warmup_edge_changes.append(
				{
					"kind": int(request.kind),
					"requested_changes": int(request.first.shape[0]),
					"edge_count_before": before_edges,
					"edge_count_after": after_edges,
					"effective_edge_delta": after_edges - before_edges if request.kind != wire.Update.FEATURES else 0,
					"effective_edge_changes": abs(after_edges - before_edges)
					if request.kind != wire.Update.FEATURES
					else 0,
				}
			)
			checks.append(check())
			logging.info("Warmup %d complete for %s", index + 1, method)
		# Build the immutable workload and final oracle before the measured window.
		# Sparse pending rows preserve repeated feature updates without cloning N*F.
		phase(args, method, repetition, "trace_preparation")
		preparing = perf_counter()
		graph = graph.clone()
		trace, pending, trace_bytes = [], {}, 0
		measured_edge_changes = []
		for index in range(args.updates):
			request = make_update(
				graph,
				features,
				spec,
				index + args.warmup,
				changes,
				generator,
				f"measured-{index}",
				workload=args.workload,
				pending=pending,
			)
			encoded = request.SerializeToString(deterministic=True)
			trace_bytes += len(encoded)
			if len(encoded) > 60 * 1024**2 or trace_bytes > 256 * 1024**2:
				raise ValueError("Workload exceeds the bounded request trace budget")
			digest.update(encoded)
			measured_edge_changes.append(edge_change_record(graph, request, mirror=True))
			nodes = torch.randint(graph.num_nodes(), (args.queries,), generator=generator)
			if args.queries:
				digest.update(nodes.numpy().tobytes())
			trace.append((request, _to_wire(nodes)))
		trace_preparation_seconds = perf_counter() - preparing
		del pending
		recording = RecordedWorkload(
			spec,
			input_identity,
			args.source_fingerprint,
			tuple(warmup_requests),
			tuple(trace),
			verification_nodes,
			args.initialization,
			args.seed + repetition,
			repetition,
			args.atol,
			args.rtol,
		)
		if recording.digest != digest.hexdigest():
			raise AssertionError("Recorded workload differs from the measured request sequence")
		recording.save(args.output / f"{name}-{method}-{repetition}-workload.pt")
		profiler = stack.enter_context(RuntimeProfiler(runtime, network_interfaces=args.network_interfaces))
		phase(args, method, repetition, "measured")
		profiler.start()
		rpc_timings.clear()
		updates, queries = [], []
		started = perf_counter()
		for index, (request, nodes) in enumerate(trace):
			updates.append(execute_update(stub, request, args.timeout))
			logging.info("Update %d: method=%s response_seconds=%.6f", index + 1, method, updates[-1]["client_seconds"])
			if not args.queries:
				continue
			before = perf_counter()
			response = stub.Query(wire.QueryRequest(nodes=nodes), timeout=args.timeout)
			assert response.version == updates[-1]["version"]
			queries.append(perf_counter() - before)
		elapsed = perf_counter() - started
		profile = profiler.stop()
		phase(args, method, repetition, "final_verification")
		rpc_report = rpc_timings.report()
		changed_nodes = []
		for request, _ in trace:
			changed_nodes.append(_from_wire(request.first, 64 * 1024**2))
			if request.kind != wire.Update.FEATURES:
				changed_nodes.append(_from_wire(request.second, 64 * 1024**2))
		verification_nodes = targeted_sample(
			graph, verification_nodes, torch.cat(changed_nodes), depth=spec.layers, seed=args.seed + repetition
		)
		checked = check()
		checked["selection"] = "uniform rows, baseline degree extremes, modified endpoints and downstream rows"
		checks.append(checked)
		result = {
			"driver_schema": 2,
			"source_fingerprint": args.source_fingerprint,
			"identity": identity,
			"input_identity": input_identity,
			"trace_digest": digest.hexdigest(),
			"policy_identity": policy_identity,
			"workload_counts": {
				**recording.counts,
				"effective_edge_changes": {
					"warmup": warmup_edge_changes,
					"measured": measured_edge_changes,
					"total_effective_edge_delta": sum(
						item["effective_edge_delta"] for item in (*warmup_edge_changes, *measured_edge_changes)
					),
					"total_effective_edge_changes": sum(
						item["effective_edge_changes"] for item in (*warmup_edge_changes, *measured_edge_changes)
					),
				},
			},
			"trace_preparation": {"seconds": trace_preparation_seconds, "request_bytes": trace_bytes},
			"query_latency": latency_summary(queries),
			"model": name,
			"method": method,
			"repetition": repetition,
			"spec": asdict(spec),
			"nodes": graph.num_nodes(),
			"edges": graph.num_edges(),
			"world_size": placement.world_size,
			"initialization": {
				"mode": args.initialization,
				"request_rows": args.initialization_batch_size if args.initialization == "all" else None,
				"seconds": initialization_seconds,
			},
			"execution_engine": "sparse-aggregates" if aggregates else "operators" if incremental else "native",
			"topology": args.topology,
			"storage_partitions": args.storage_partitions,
			"feature_layout": {"contiguous": features.is_contiguous(), "stride": list(features.stride())},
			"anchor_budget_bytes": runtime.service._delta.state.anchor_bytes
			if runtime.service._delta is not None
			else None,
			"neighborhood_policy": {
				"fanout": policy.fanout,
				"reuse_intermediates": policy.reuse_intermediates,
				"seed": policy.seed,
			},
			"dispatch_policy": runtime.service.profile()["dispatch"],
			"deferred_policy": runtime.service.profile()["deferred"],
			"worker_feature_cache": {
				"enabled": worker_cache,
				"budget_bytes": worker_cache_bytes,
			},
			"switches": switches.as_dict(),
			"correctness": checks,
			"updates": updates,
			"query_seconds": queries,
			"window_seconds": elapsed,
			"completed_changes_per_second": sum(item["changes"] for item in updates) / elapsed,
			"profile": profile,
			"rpc": rpc_report,
			"valid": profile["valid"] and rpc_report["dropped"] == 0,
		}
		result["metrics"] = measurements(result)
		result["response_seconds"] = result["metrics"]["response_time"]["mean_seconds"]
		phase(args, method, repetition, "complete")
		return result


def phase(args, method, repetition, name):
	"""Publish boundaries outside the measured window for bounded trial supervision."""
	path = getattr(args, "phase_file", None)
	if path is not None:
		path.parent.mkdir(parents=True, exist_ok=True)
		temporary = path.with_suffix(".tmp")
		temporary.write_text(
			json.dumps({"method": method, "repetition": repetition, "phase": name, "time_unix": time.time()}) + "\n"
		)
		temporary.replace(path)


def _run(args: Options):
	placement = Placement.resolve(
		rank=args.rank, world_size=args.world_size, local_rank=args.local_rank, init_method=args.init_method
	)
	if placement.rank is None:
		raise ValueError("Launch this experiment with torchrun or explicit rank/world-size/rendezvous")
	torch.set_num_threads(args.threads)
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	base_graph = base_features = None
	if placement.rank == 0:
		args.output.mkdir(parents=True, exist_ok=False)
		args.source_fingerprint = source_fingerprint(args.output / "sources.zip")
		base_graph, base_features = load_graph(args)
		base_graph = base_graph.formats("coo")
		report: dict[str, Any] = {
			"environment": {
				"python": platform.python_version(),
				"torch": torch.__version__,
				"dgl": dgl.__version__,
				"cuda_runtime": torch.version.cuda,
			},
			"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
			"trials": [],
		}
	for method in args.methods:
		for repetition in range(
			getattr(args, "repetition_offset", 0), getattr(args, "repetition_offset", 0) + args.repetitions
		):
			models = args.models if repetition % 2 == 0 else list(reversed(args.models))
			for name in models:
				logging.info("Trial model=%s method=%s repetition=%s rank=%s", name, method, repetition, placement.rank)
				result = trial(args, placement, base_graph, base_features, name, method, repetition)
				if placement.rank == 0:
					path = args.output / f"{name}-{method}-{repetition}.json"
					with path.open("x") as stream:
						json.dump(result, stream, indent=2, allow_nan=False)
					report["trials"].append(
						{
							"file": path.name,
							"valid": result["valid"],
							"model": name,
							"method": method,
							"fanout": result["neighborhood_policy"]["fanout"],
							"throughput": result["completed_changes_per_second"],
						}
					)
	if placement.rank == 0:
		report["comparisons"] = []
		for name in args.models:
			groups = {
				method: [
					json.loads((args.output / item["file"]).read_text())
					for item in report["trials"]
					if item["model"] == name and item["method"] == method
				]
				for method in args.methods
			}
			if NRT_INC in groups:
				for baseline in groups.keys() - {NRT_INC}:
					report["comparisons"].append(
						{
							"model": name,
							"baseline": baseline,
							"candidate": NRT_INC,
							**paired_comparison(groups[baseline], groups[NRT_INC], metric="response_seconds"),
						}
					)
		with (args.output / "summary.json").open("x") as stream:
			json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
	app()
