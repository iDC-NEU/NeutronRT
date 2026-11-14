"""Check local CPU storage/model processes against full DGL inference."""

import json
import subprocess
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from scripts.validate.inference import reference


@torch.inference_mode()
def check_sum_rows(service, model, graph, features, nodes):
	"""Compare selected retained sums with a bounded, independent CPU edge loop."""
	nodes = nodes.unique(sorted=True)
	sources, targets = graph.in_edges(nodes)
	positions = torch.searchsorted(nodes, targets)
	current = features
	errors = []
	for index in range(model.num_layers):
		service.ensure(nodes, index + 1)
		layer = model.layers[index]
		expected = torch.zeros((nodes.numel(), layer.fc_neigh.out_features), dtype=features.dtype)
		for start in range(0, sources.numel(), 8192):
			end = start + 8192
			expected.index_add_(0, positions[start:end], layer.fc_neigh(current[sources[start:end]]))
		maximum = 0.0
		for group in service.aggregate_catalog.selections(index + 1, nodes, service.catalog.version):
			with ExitStack() as leases:
				actual = service.remote_plane.read_rows(group.key, group.rows, leases)
				wanted = expected[group.positions]
				torch.testing.assert_close(actual, wanted, rtol=2e-4, atol=3e-5)
				maximum = max(maximum, float((actual - wanted).abs().max()) if actual.numel() else 0.0)
		errors.append(maximum)
		if index + 1 < model.num_layers:
			current = model(index, graph, current)
	return errors


def validate(args, name, partitions, workers, concurrent):
	products = args.dataset == "ogbn-products"
	graph = load_graph_data(Path(args.data), add_inverse_edges=products)
	features = graph.ndata.pop("feat")
	expected_shape = (2449029, 100) if products else (169343, 128)
	if tuple(features.shape) != expected_shape:
		raise ValueError(f"Expected official {args.dataset} features with shape {expected_shape}")
	query = torch.cat(
		(
			torch.arange(args.changes),
			torch.randint(graph.num_nodes(), (128,), generator=torch.Generator().manual_seed(71)),
		)
	)
	spec = ModelSpec(
		name,
		features.shape[1],
		64,
		47 if products else 40,
		checkpoint=str(args.checkpoint) if name == "sage" and args.checkpoint else None,
	)
	model = spec.build()
	switches = RuntimeFeatures(
		distributed_storage=True,
		pipeline=True,
		concurrent_compute=concurrent,
		gather_inputs=True,
		bounded_staleness=not args.retain_aggregates,
		retain_aggregates=args.retain_aggregates,
		aggregate_only=args.aggregate_only,
		incremental_aggregates=args.incremental_aggregates,
		row_refresh=args.row_refresh,
	)
	start = perf_counter()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu"] * workers,
		switches=switches,
		storage_partitions=partitions,
		storage_directory=str(args.topology_root),
		worker_bytes=args.worker_bytes,
		storage_bytes=args.storage_bytes,
		batch_size=args.batch_size,
		idle_flush_seconds=60,
		timeout=60,
		delta_max_steps=args.delta_max_steps,
		delta_anchor_bytes=args.delta_anchor_bytes,
	) as runtime:
		startup = perf_counter() - start
		service = runtime.service
		owner = service.remote_plane.storage.owner
		pointer = features.data_ptr()
		records = []

		def check(operation):
			with service.query(query, owner) as actual:
				expected = reference(model, graph, features, query)
				torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
				record = {
					"operation": operation,
					"version": service.catalog.version,
					"max_abs_error": float((actual - expected).abs().max()),
					"prediction_disagreement": float((actual.argmax(1) != expected.argmax(1)).float().mean()),
				}
				if args.retain_aggregates:
					record["aggregate_max_abs_error"] = check_sum_rows(service, model, graph, features, query)
				records.append(record)
			assert service.features.data_ptr() == pointer and not torch.cuda.is_initialized()

		check("cold-query")
		if args.incremental_aggregates:
			# Establish equal warm coverage before testing the sparse update path.
			service.ensure(torch.arange(graph.num_nodes()))
		with closing(UpdateAdmission(service)) as admission:
			changed = torch.arange(args.changes)
			admission.sync(UpdateKind.FEATURES, changed, features[changed] + 0.25)
			check("features")
			u, v = changed, torch.flip(changed, (0,))
			admission.sync(UpdateKind.INSERT, u, v)
			graph.add_edges(u, v)
			check("insert")
			admission.sync(UpdateKind.DELETE, u, v)
			_, _, ids = graph.edge_ids(u, v, return_uv=True)
			graph.remove_edges(ids.unique())
			check("delete")
			first = admission.submit(UpdateKind.FEATURES, changed, torch.ones(args.changes, features.shape[1]))
			second = admission.submit(UpdateKind.FEATURES, changed, torch.full((args.changes, features.shape[1]), 2.0))
			admission.sync(UpdateKind.FEATURES, changed, torch.full((args.changes, features.shape[1]), 3.0))
			assert first.result(120).input_version == 4 and second.result(120).input_version == 5
			check("consecutive-features" if args.retain_aggregates else "merged-features")
		assert service.counters.updates == 6 and service.counters.input_changes == args.changes * 6
		if args.incremental_aggregates:
			assert service.counters.delta_tasks > 0
		stats = runtime.topology.storage.stats()
		assert all(not value["cuda_initialized"] and not value["response_buffers"] for value in stats.values())
		return {
			"valid": True,
			"dataset": args.dataset,
			"nodes": graph.num_nodes(),
			"edges": graph.num_edges(),
			"checkpoint": str(args.checkpoint) if args.checkpoint else None,
			"accuracy_scope": "Fresh inference agreement; untrained weights when no checkpoint supplied",
			"model": name,
			"storage_partitions": partitions,
			"compute_workers": workers,
			"concurrent_compute": concurrent,
			"retain_aggregates": args.retain_aggregates,
			"aggregate_only": args.aggregate_only,
			"incremental_aggregates": args.incremental_aggregates,
			"row_refresh": args.row_refresh,
			"delta_max_steps": args.delta_max_steps,
			"delta_anchor_bytes": args.delta_anchor_bytes,
			"worker_bytes": args.worker_bytes,
			"batch_size": args.batch_size,
			"startup_seconds": startup,
			"checks": records,
			"storage": stats,
			"counters": asdict(service.counters),
		}


DatasetChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DatasetChoice", {str(value).replace("-", "_").upper(): value for value in ["ogbn-arxiv", "ogbn-products"]}
)
ModelsChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"ModelsChoice", {str(value).replace("-", "_").upper(): value for value in ["sage", "gat", "gcn"]}
)


_DEFAULT_DATASET = DatasetChoice("ogbn-arxiv")


@dataclass
class Options:
	data: Path
	dataset: str
	checkpoint: Path | None
	topology_root: Path
	output: Path
	models: list[str]
	changes: int
	threads: int
	worker_bytes: int
	storage_bytes: int
	batch_size: int
	quick: bool
	retain_aggregates: bool
	aggregate_only: bool
	incremental_aggregates: bool
	delta_max_steps: int
	row_refresh: bool
	delta_anchor_bytes: int


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	dataset: Annotated[DatasetChoice, typer.Option("--dataset")] = _DEFAULT_DATASET,
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	topology_root: Annotated[Path, typer.Option("--topology-root")] = Path("data/evaluation/topology"),
	models: Annotated[list[ModelsChoice] | None, typer.Option("--models")] = None,
	changes: Annotated[int, typer.Option("--changes")] = 16,
	threads: Annotated[int, typer.Option("--threads")] = 4,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 512 * 1024**2,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 256 * 1024**2,
	batch_size: Annotated[int, typer.Option("--batch-size")] = 8192,
	quick: Annotated[
		bool, typer.Option("--quick", help="Only the three-storage/two-compute concurrent setting")
	] = False,
	retain_aggregates: Annotated[
		bool, typer.Option("--retain-aggregates", help="Validate exact retained sums; use --models sage")
	] = False,
	aggregate_only: Annotated[
		bool, typer.Option("--aggregate-only", help="Drop h under pressure and reconstruct from sums")
	] = False,
	incremental_aggregates: Annotated[
		bool, typer.Option("--incremental-aggregates", help="Update projected sums over sparse routes")
	] = False,
	delta_max_steps: Annotated[int, typer.Option("--delta-max-steps")] = 32,
	row_refresh: Annotated[
		bool, typer.Option("--row-refresh", help="Refresh expired rows within sparse target chunks")
	] = False,
	delta_anchor_bytes: Annotated[int, typer.Option("--delta-anchor-bytes")] = 256 * 1024**2,
):
	args = Options(
		**{
			**locals(),
			"dataset": dataset.value,
			"models": [item.value for item in models] if models is not None else ["sage", "gat", "gcn"],
		}
	)
	if not 1 <= args.changes <= 128 or args.threads < 1:
		raise typer.BadParameter("Use 1..128 changes and a positive CPU thread count")
	if min(args.worker_bytes, args.batch_size) < 1 or args.storage_bytes < 0:
		raise typer.BadParameter("Use positive worker/batch budgets and a nonnegative storage budget")
	if args.aggregate_only and not args.retain_aggregates:
		raise typer.BadParameter("Aggregate-only validation requires --retain-aggregates")
	if args.incremental_aggregates and not args.retain_aggregates:
		raise typer.BadParameter("Sparse validation requires --retain-aggregates")
	if args.delta_max_steps < 1 or args.delta_anchor_bytes < 0:
		raise typer.BadParameter("Use a positive delta refresh interval and nonnegative anchor budget")
	if args.row_refresh and not args.incremental_aggregates:
		raise typer.BadParameter("Row refresh requires --incremental-aggregates")
	if args.output.exists():
		raise typer.BadParameter("Preserve previous validation artifacts; choose a new output path")
	if args.retain_aggregates and args.models != ["sage"]:
		raise typer.BadParameter("Retained sums currently require --models sage")
	if args.dataset == "ogbn-products" and args.models != ["sage"]:
		raise typer.BadParameter("Establish products model logic with --models sage before extending other models")
	torch.set_num_threads(args.threads)
	settings = [(3, 2, not args.retain_aggregates)] if args.quick else [(1, 1, False), (3, 1, False), (3, 2, False)]
	if not args.quick and not args.retain_aggregates:
		settings.append((3, 2, True))
	results = []
	for name in args.models:
		for partitions, workers, concurrent in settings:
			result = validate(args, name, partitions, workers, concurrent)
			results.append(result)
			args.output.parent.mkdir(parents=True, exist_ok=True)
			args.output.write_text(
				json.dumps(
					{
						"scope": "Local CPU processes; no communication-failure injection or remote scaling",
						"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
						"results": results,
					},
					indent=2,
				)
				+ "\n"
			)
			print(json.dumps(result), flush=True)


if __name__ == "__main__":
	app()
