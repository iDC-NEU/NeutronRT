"""Stream long sparse-update traces and check fresh outputs without logit history."""

import json
import subprocess
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import pandas as pd
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.models.spec import ModelSpec
from lib.profiling.accuracy import query_error
from lib.profiling.experiment import make_trace, trace_digest
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from scripts.evaluate.profile import _apply_reference
from scripts.validate.inference import reference
from scripts.validate.storage import check_sum_rows


def observe_trace(service, model, graph, trace, query, labels, *, every, emit):
	"""Keep metrics only; authoritative features are shared with the CPU oracle."""
	if type(every) is not int or every < 1:
		raise ValueError("Observation interval must be a positive integer")
	records = []

	def observe():
		with service.query(query, service.remote_plane.storage.owner) as actual:
			fresh = reference(model, graph, service.features, query)
			record = {
				"version": service.catalog.version,
				"valid": bool(torch.isclose(actual, fresh, rtol=2e-4, atol=3e-5).all()),
				**query_error(actual, fresh, labels=labels),
			}
		# Sum checks use selected rows and bounded edge-vector scratch. No
		# previous full output or whole-graph aggregate matrix is retained.
		record["aggregate_max_abs_error"] = check_sum_rows(service, model, graph, service.features, query)
		record["counters"] = asdict(service.counters)
		records.append(record)
		emit(record)

	observe()
	with closing(UpdateAdmission(service)) as admission:
		for index, update in enumerate(trace, start=1):
			admission.sync(UpdateKind(update.kind), update.first, update.second)
			if update.kind != "features":
				_apply_reference(graph, service.features, [update])
			if index % every == 0 or index == len(trace):
				observe()
	return records


@dataclass
class Options:
	data: Path
	checkpoint: Path
	output: Path
	refresh_steps: list[int]
	row_refresh: bool
	aggregate_only: bool
	updates: int
	changes: int
	pool_size: int
	observe_every: int
	threads: int
	storage_partitions: int
	workers: int
	worker_bytes: int
	storage_bytes: int
	topology_root: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	checkpoint: Annotated[Path, typer.Option("--checkpoint")],
	output: Annotated[Path, typer.Option("--output")],
	refresh_steps: Annotated[list[int] | None, typer.Option("--refresh-steps")] = None,
	row_refresh: Annotated[bool, typer.Option("--row-refresh")] = False,
	aggregate_only: Annotated[bool, typer.Option("--aggregate-only")] = False,
	updates: Annotated[int, typer.Option("--updates")] = 128,
	changes: Annotated[int, typer.Option("--changes")] = 16,
	pool_size: Annotated[int, typer.Option("--pool-size")] = 128,
	observe_every: Annotated[int, typer.Option("--observe-every")] = 16,
	threads: Annotated[int, typer.Option("--threads")] = 4,
	storage_partitions: Annotated[int, typer.Option("--storage-partitions")] = 3,
	workers: Annotated[int, typer.Option("--workers")] = 2,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 32 * 1024**2,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 256 * 1024**2,
	topology_root: Annotated[Path, typer.Option("--topology-root")] = Path("data/evaluation/topology"),
):
	args = Options(**{**locals(), "refresh_steps": refresh_steps if refresh_steps is not None else [2, 8, 32]})
	_run(args)


def _run(args: Options) -> None:
	if (
		min(
			args.updates,
			args.changes,
			args.observe_every,
			args.threads,
			args.workers,
			args.storage_partitions,
			args.worker_bytes,
			*args.refresh_steps,
		)
		< 1
		or args.storage_bytes < 0
	):
		raise typer.BadParameter("Counts, refresh steps and worker bytes must be positive; storage bytes nonnegative")
	if args.pool_size < args.changes:
		raise typer.BadParameter("The hotspot pool must contain every changed row")
	if args.output.exists():
		raise typer.BadParameter("Preserve previous artifacts; choose a new output path")
	torch.set_num_threads(args.threads)
	report: dict[str, Any] = {
		"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
		"scope": "Local CPU processes, fixed mean GraphSAGE, full official arXiv test split at observed versions",
		"timing_scope": "Includes fresh CPU oracles; use matched profiler runs for performance comparisons",
		"settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
		"runs": [],
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	for steps in args.refresh_steps:
		graph = load_graph_data(Path(args.data))
		features = graph.ndata.pop("feat")
		if tuple(features.shape) != (169343, 128):
			raise ValueError("Establish sparse-update accuracy on official ogbn-arxiv first")
		query = torch.from_numpy(
			pd.read_csv(args.data.parent / "split/time/test.csv.gz", header=None).to_numpy().reshape(-1)
		).long()
		labels = torch.from_numpy(
			pd.read_csv(args.data / "node-label.csv.gz", header=None).to_numpy().reshape(-1)
		).long()[query]
		if query.unique().numel() != 48603:
			raise ValueError("Expected all 48,603 official test nodes")
		trace = make_trace(features, updates=args.updates, changes=args.changes, pool_size=args.pool_size, mixed=True)
		spec = ModelSpec("sage", 128, 64, 40, checkpoint=str(args.checkpoint))
		run: dict[str, Any] = {"refresh_steps": steps, "trace_digest": trace_digest(trace), "observations": []}
		report["runs"].append(run)
		start = perf_counter()
		with ManagedRuntime(
			graph,
			features,
			spec,
			["cpu"] * args.workers,
			switches=RuntimeFeatures(
				pipeline=True,
				gather_inputs=True,
				retain_aggregates=True,
				aggregate_only=args.aggregate_only,
				incremental_aggregates=True,
				row_refresh=args.row_refresh,
				distributed_storage=True,
			),
			storage_partitions=args.storage_partitions,
			storage_directory=str(args.topology_root),
			worker_bytes=args.worker_bytes,
			storage_bytes=args.storage_bytes,
			batch_size=2048,
			delta_max_steps=steps,
			timeout=60,
		) as runtime:
			runtime.service.ensure(torch.arange(graph.num_nodes()))

			def emit(record, run=run, steps=steps):
				run["observations"].append(record)
				args.output.write_text(json.dumps(report, indent=2) + "\n")
				print(json.dumps({"refresh_steps": steps, **record}), flush=True)

			observe_trace(
				runtime.service, spec.build(), graph, trace, query, labels, every=args.observe_every, emit=emit
			)
			run["valid"] = all(item["valid"] for item in run["observations"])
			run["seconds_including_oracles"] = perf_counter() - start
			assert runtime.service.counters.delta_tasks > 0 and not torch.cuda.is_initialized()
		args.output.write_text(json.dumps(report, indent=2) + "\n")
	if not all(run["valid"] for run in report["runs"]):
		raise SystemExit("Sparse drift exceeded output tolerance; inspect retained accuracy observations")


if __name__ == "__main__":
	app()
