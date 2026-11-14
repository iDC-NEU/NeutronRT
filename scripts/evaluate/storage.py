"""Measure local topology partitions separately from model computation."""

import json
import statistics
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import dgl
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.recompute import compile_block
from lib.profiling.storage import start, stop
from lib.storage.topology import ShardedTopology


def block_edges(block, nodes):
	u, v = block.edges()
	return (block.srcdata[dgl.NID][u] * nodes + block.dstdata[dgl.NID][v]).sort().values


def read(topology, targets, frontier):
	block = compile_block(topology, targets)
	if hasattr(topology, "expand"):
		expanded = topology.expand(frontier, targets)
	else:
		expanded = torch.cat((frontier, targets, topology.out_edges(frontier)[1])).unique(sorted=True)
	return block, expanded, topology.in_degrees(targets), topology.out_degrees(targets)


def check(actual, expected, nodes):
	torch.testing.assert_close(block_edges(actual[0], nodes), block_edges(expected[0], nodes))
	for value, reference in zip(actual[1:], expected[1:], strict=True):
		torch.testing.assert_close(value, reference)


DatasetChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DatasetChoice", {str(value).replace("-", "_").upper(): value for value in ["ogbn-arxiv", "ogbn-products"]}
)


_DEFAULT_DATASET = DatasetChoice("ogbn-arxiv")


@dataclass
class Options:
	root: Path
	dataset: str
	partitions: list[int]
	trials: int
	nodes: int
	threads: int
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	root: Annotated[Path, typer.Option("--root")],
	output: Annotated[Path, typer.Option("--output")],
	dataset: Annotated[DatasetChoice, typer.Option("--dataset")] = _DEFAULT_DATASET,
	partitions: Annotated[list[int] | None, typer.Option("--partitions")] = None,
	trials: Annotated[int, typer.Option("--trials")] = 5,
	nodes: Annotated[int, typer.Option("--nodes")] = 2048,
	threads: Annotated[int, typer.Option("--threads")] = 1,
):
	args = Options(
		**{**locals(), "dataset": dataset.value, "partitions": partitions if partitions is not None else [1, 2, 4]}
	)
	_run(args)


def _run(args: Options) -> None:
	if min([*args.partitions, args.trials, args.nodes, args.threads]) < 1 or 1 not in args.partitions:
		raise typer.BadParameter("Counts must be positive; include one storage partition")
	if args.output.exists():
		raise typer.BadParameter("Choose a new report path to preserve previous trials")
	torch.set_num_threads(args.threads)
	graph = load_graph_data(
		Path(str(args.root / args.dataset.removeprefix("ogbn-") / "raw")),
		add_inverse_edges=args.dataset == "ogbn-products",
	)
	del graph.ndata["feat"]
	graph.create_formats_()
	if args.nodes > graph.num_nodes():
		raise typer.BadParameter("Requested rows exceed graph nodes")
	generator = torch.Generator().manual_seed(71)
	targets = torch.randperm(graph.num_nodes(), generator=generator)[: args.nodes]
	frontier = torch.randperm(graph.num_nodes(), generator=generator)[: args.nodes]
	expected = read(graph, targets, frontier)
	u = torch.randint(graph.num_nodes(), (128,), generator=generator)
	v = torch.randint(graph.num_nodes(), (128,), generator=generator)
	absent = ~graph.has_edges_between(u, v)
	u, v = u[absent], v[absent]
	if not u.numel():
		raise ValueError("This validation trace needs initially absent edges")
	report = {
		"dataset": args.dataset,
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"scope": "Local topology storage only; no model computation, GPU traffic or communication-failure injection",
		"warm_cache_note": "Export and warmup precede measurements; OS caches are not flushed",
		"comparison_note": (
			"Settings run sequentially. Ratios are descriptive medians, without statistical speedup verdicts. "
			"Mutation times have no DGL baseline."
		),
		"settings": [],
		"dgl_reads": [],
	}
	for trial in range(args.trials + 1):
		begin = perf_counter()
		actual = read(graph, targets, frontier)
		seconds = perf_counter() - begin
		check(actual, expected, graph.num_nodes())
		if trial:
			report["dgl_reads"].append({"seconds": seconds})
	for partitions in args.partitions:
		with ShardedTopology(graph, partitions, directory=str(args.root / "topology"), profiling=True) as topology:
			check(read(topology, targets, frontier), expected, graph.num_nodes())
			before, samplers = start(topology)
			trials = []
			try:
				for _ in range(args.trials):
					begin = perf_counter()
					actual = read(topology, targets, frontier)
					seconds = perf_counter() - begin
					check(actual, expected, graph.num_nodes())
					begin = perf_counter()
					topology.add_edges(u, v)
					insert_seconds = perf_counter() - begin
					assert topology.num_edges() == graph.num_edges() + u.numel()
					assert topology.has_edges_between(u, v).all()
					begin = perf_counter()
					topology.delete_pairs(u, v)
					delete_seconds = perf_counter() - begin
					assert topology.num_edges() == graph.num_edges() and not topology.has_edges_between(u, v).any()
					trials.append(
						{"seconds": seconds, "insert_seconds": insert_seconds, "delete_seconds": delete_seconds}
					)
			finally:
				profile = stop(topology, before, samplers)
			check(read(topology, targets, frontier), expected, graph.num_nodes())
			report["settings"].append(
				{
					"partitions": partitions,
					"trials": trials,
					"profile": profile,
					"vs_dgl_read_ratio": statistics.median(row["seconds"] for row in report["dgl_reads"])
					/ statistics.median(row["seconds"] for row in trials),
				}
			)
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(json.dumps(report, indent=2) + "\n")
		print(json.dumps({"partitions": partitions, "valid": True, "trials": trials}), flush=True)
	baseline = next(setting["trials"] for setting in report["settings"] if setting["partitions"] == 1)
	for setting in report["settings"]:
		setting["vs_one_partition_ratio"] = statistics.median(row["seconds"] for row in baseline) / statistics.median(
			row["seconds"] for row in setting["trials"]
		)
	args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	app()
