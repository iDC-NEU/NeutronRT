"""Check exact aggregate-only GraphSAGE storage on CPU before runtime integration.

Reports logical retained tensor sizes and selected-row reconstruction latency.
This is not distributed serving throughput or a GPU performance measurement.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

import pandas as pd
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.deltas import NeighborUpdate
from lib.models.aggregates import reconstruct_sage
from lib.models.spec import ModelSpec
from lib.profiling.accuracy import query_error


def _bytes(tensor):
	return tensor.numel() * tensor.element_size()


@torch.inference_mode()
def evaluate(graph, features, model, queries, labels, *, repetitions=5, query_rows=2048, chunk_edges=8192):
	if model.num_layers != 2 or model.model_name != "sage" or model.training:
		raise ValueError("This checkpoint comparison requires eval-mode two-layer GraphSAGE")
	if repetitions < 1 or query_rows < 1 or chunk_edges < 1 or not queries.numel():
		raise ValueError("Use positive query/repetition/chunk counts and nonempty evaluation rows")
	first = model(0, graph, features)
	expected = model(1, graph, first)
	layer_inputs = (features, first)
	embedding_bytes = _bytes(first) + _bytes(expected)
	degree = graph.in_degrees()
	sources, destinations = graph.edges()
	generator = torch.Generator().manual_seed(81)
	batches = [queries[torch.randint(queries.numel(), (query_rows,), generator=generator)] for _ in range(8)]
	report = {
		"scope": "CPU aggregate representation/reconstruction only; not integrated distributed serving",
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"features_bytes": _bytes(features),
		"current_embedding_only_bytes": embedding_bytes,
		"notes": [
			"Logical retained tensor bytes exclude model/topology/metadata, allocator overhead and temporary scratch.",
			"Each layer is charged its degree vector; sharing across equal anchors could reduce this later.",
			"Read timings include selection and transforms, with warm CPU memory; no transport or GPU time.",
			"Full DGL reference embeddings exist for validation only; an aggregate-only serving engine is pending.",
		],
		"settings": [],
	}
	for projected in (False, True):
		states = []
		for index, inputs in enumerate(layer_inputs):
			values = model.layers[index].fc_neigh(inputs) if projected else inputs
			sums = torch.zeros((graph.num_nodes(), values.shape[1]), dtype=values.dtype)
			for start in range(0, sources.numel(), chunk_edges):
				end = start + chunk_edges
				sums.index_add_(0, destinations[start:end], values[sources[start:end]])
			states.append(NeighborUpdate(sums, degree))
		retained = sum(_bytes(state.sums) + _bytes(state.degrees) for state in states)
		errors = []
		for start in range(0, queries.numel(), query_rows):
			nodes = queries[start : start + query_rows]
			actual = reconstruct_sage(
				model,
				features[nodes],
				[NeighborUpdate(state.sums[nodes], state.degrees[nodes]) for state in states],
				projected=projected,
			)
			errors.append(actual)
		actual = torch.cat(errors)
		torch.testing.assert_close(actual, expected[queries], atol=3e-5, rtol=2e-4)
		error = query_error(actual, expected[queries], labels=labels[queries])
		cached_times, reconstructed_times = [], []

		def read(nodes, reconstruct, states=states, projected=projected):
			if not reconstruct:
				return expected[nodes]
			return reconstruct_sage(
				model,
				features[nodes],
				[NeighborUpdate(state.sums[nodes], state.degrees[nodes]) for state in states],
				projected=projected,
			)

		# Equal warmup and alternating ordering; raw measurements remain in the report.
		for mode in (False, True):
			for nodes in batches:
				read(nodes, mode)
		for repetition in range(repetitions):
			for mode in (False, True) if repetition % 2 == 0 else (True, False):
				started = perf_counter()
				for nodes in batches:
					read(nodes, mode)
				seconds = perf_counter() - started
				(reconstructed_times if mode else cached_times).append(seconds / len(batches))
		report["settings"].append(
			{
				"projected": projected,
				"valid": True,
				"aggregate_only_bytes": retained,
				"aggregate_and_embeddings_bytes": retained + embedding_bytes,
				"saved_embedding_bytes": embedding_bytes,
				"state_reduction_fraction": embedding_bytes / (retained + embedding_bytes),
				"including_features_reduction_fraction": embedding_bytes
				/ (retained + embedding_bytes + _bytes(features)),
				"query_rows": query_rows,
				"raw_cached_read_seconds": cached_times,
				"raw_reconstructed_read_seconds": reconstructed_times,
				"median_cached_read_seconds": statistics.median(cached_times),
				"median_reconstructed_read_seconds": statistics.median(reconstructed_times),
				"accuracy": error,
			}
		)
	return report


@dataclass
class Options:
	data: Path
	checkpoint: Path
	output: Path
	threads: int
	repetitions: int
	query_rows: int


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	checkpoint: Annotated[Path, typer.Option("--checkpoint")],
	output: Annotated[Path, typer.Option("--output")],
	threads: Annotated[int, typer.Option("--threads")] = 1,
	repetitions: Annotated[int, typer.Option("--repetitions")] = 5,
	query_rows: Annotated[int, typer.Option("--query-rows")] = 2048,
):
	args = Options(**locals())
	if args.output.exists() or min(args.threads, args.repetitions, args.query_rows) < 1:
		raise typer.BadParameter("Choose positive counts and a new output path")
	torch.set_num_threads(args.threads)
	graph = load_graph_data(Path(args.data))
	features = graph.ndata.pop("feat")
	if tuple(features.shape) != (169343, 128):
		raise typer.BadParameter("Establish aggregate-only correctness on official ogbn-arxiv first")
	queries = torch.tensor(pd.read_csv(args.data.parent / "split/time/test.csv.gz", header=None).to_numpy().reshape(-1))
	labels = torch.tensor(pd.read_csv(args.data / "node-label.csv.gz", header=None).to_numpy().reshape(-1))
	model = ModelSpec("sage", 128, 64, 40, checkpoint=str(args.checkpoint)).build()
	report = evaluate(graph, features, model, queries, labels, repetitions=args.repetitions, query_rows=args.query_rows)
	report.update(
		{"dataset": "ogbn-arxiv", "accuracy_scope": "Full official test split", "checkpoint": str(args.checkpoint)}
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(report, indent=2) + "\n")
	print(json.dumps(report), flush=True)


if __name__ == "__main__":
	app()
