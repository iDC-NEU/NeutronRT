"""Measure native threads over existing graph/embedding tensor storage."""

import json
import statistics
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter, process_time
from typing import Annotated, Callable

import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.native import NativeDeltaAccumulator, compile_native

DatasetChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DatasetChoice", {str(value).replace("-", "_").upper(): value for value in ["ogbn-arxiv", "ogbn-products"]}
)


_DEFAULT_DATASET = DatasetChoice("ogbn-arxiv")


@dataclass
class Options:
	dataset: str
	root: str
	output: Path
	changes: list[int]
	threads: list[int]
	trials: int


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	root: Annotated[str, typer.Option("--root")],
	output: Annotated[Path, typer.Option("--output")],
	dataset: Annotated[DatasetChoice, typer.Option("--dataset")] = _DEFAULT_DATASET,
	changes: Annotated[list[int] | None, typer.Option("--changes")] = None,
	threads: Annotated[list[int] | None, typer.Option("--threads")] = None,
	trials: Annotated[int, typer.Option("--trials")] = 5,
):
	args = Options(
		**{
			**locals(),
			"dataset": dataset.value,
			"changes": changes if changes is not None else [1000, 10000],
			"threads": threads if threads is not None else [1, 2, 4, 8],
		}
	)
	_run(args)


def _run(args: Options) -> None:
	if min(args.changes + args.threads + [args.trials]) < 1:
		raise typer.BadParameter("Sizes, trials, and thread counts must be positive")
	if 1 not in args.threads:
		raise typer.BadParameter("Include one native thread to measure scaling with the same algorithm")
	torch.set_num_threads(1)
	raw = Path(args.root) / args.dataset.removeprefix("ogbn-") / "raw"
	graph = load_graph_data(Path(raw), add_inverse_edges=args.dataset == "ogbn-products")
	graph.create_formats_()
	current = graph.ndata.pop("feat")
	output = torch.zeros(current.shape, dtype=torch.float64)
	report = {
		"dataset": args.dataset,
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"dimensions": current.shape[1],
		"input_strides": list(current.stride()),
		"torch_threads": 1,
		"scope": "Native threads share graph/current/previous/output storage; no full graph or feature copy in kernels",
		"comparison": "Native counts share a CSC pull algorithm; torch scatter uses different edge traversal",
		"batches": [],
	}
	with tempfile.TemporaryDirectory(prefix="neutronrt-threads-") as directory:
		native = NativeDeltaAccumulator(graph, compile_native(directory))
		for size in args.changes:
			if size > graph.num_nodes():
				raise ValueError("Requested changes exceed graph nodes")
			generator = torch.Generator().manual_seed(size + 42)
			changed = torch.randperm(graph.num_nodes(), generator=generator)[:size].sort().values
			previous = current[changed] - 0.25
			u, v = graph.out_edges(changed)
			targets = v.unique()
			positions = torch.searchsorted(changed, u)

			def scatter(changed=changed, previous=previous, v=v, positions=positions):
				delta = current[changed].double() - previous.double()
				for begin in range(0, v.numel(), 8192):
					output.index_add_(0, v[begin : begin + 8192], delta[positions[begin : begin + 8192]])

			variants: dict[str, Callable[..., object]] = {"torch_scatter": scatter}
			for threads in args.threads:
				variants[f"native_{threads}"] = (
					lambda threads=threads, previous=previous, changed=changed, targets=targets: native.apply(
						current, previous, changed, targets, output, threads=threads
					)
				)
			# Previous batches leave accumulated output; build each oracle from zero.
			output.zero_()
			scatter()
			expected = output[targets].clone()
			pointers = current.data_ptr(), output.data_ptr(), native.offsets.data_ptr(), native.sources.data_ptr()
			records: dict[str, list[dict[str, float]]] = {key: [] for key in variants}
			for trial in range(args.trials + 1):
				order = list(variants)
				order = order[trial % len(order) :] + order[: trial % len(order)]
				for name in order:
					output.index_fill_(0, targets, 0)
					cpu_start = process_time()
					start = perf_counter()
					variants[name]()
					wall = perf_counter() - start
					cpu = process_time() - cpu_start
					torch.testing.assert_close(output[targets], expected, rtol=1e-10, atol=1e-10)
					assert pointers == (
						current.data_ptr(),
						output.data_ptr(),
						native.offsets.data_ptr(),
						native.sources.data_ptr(),
					)
					if trial:
						records[name].append({"wall_seconds": wall, "cpu_seconds": cpu, "cpu_to_wall": cpu / wall})
			medians = {key: statistics.median(row["wall_seconds"] for row in rows) for key, rows in records.items()}
			batch = {
				"changed_nodes": size,
				"affected_targets": targets.numel(),
				"changed_edges": u.numel(),
				"pull_edges_visited": int(graph.in_degrees(targets).sum()),
				"median_seconds": medians,
				"speedup_vs_torch": {key: medians["torch_scatter"] / value for key, value in medians.items()},
				"native_thread_scaling": {
					key: medians["native_1"] / value for key, value in medians.items() if key.startswith("native_")
				},
				"native_input_storage_unchanged": True,
				"runs": records,
			}
			report["batches"].append(batch)
			print(json.dumps({key: value for key, value in batch.items() if key != "runs"}), flush=True)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	app()
