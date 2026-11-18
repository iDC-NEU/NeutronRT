"""Replay a recorded workload using native DGL full or sampled recomputation."""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from benchmarks.streaming import run
from lib.dataset.inference import load_graph
from lib.profiling.accuracy import load_test_labels
from lib.profiling.workload import RecordedWorkload


class Method(StrEnum):
	FULL = "RTEC-Full"
	NS = "RTEC-NS"
	UER = "RTEC-UER"


app = typer.Typer(help=__doc__, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option(exists=True)],
	workload: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(help="New directory for a source snapshot and measured trial")],
	checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	labels: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	test_split: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	method: Annotated[Method, typer.Option()] = Method.NS,
	fanout: Annotated[int, typer.Option(min=1, help="Positive fanout at every layer for RTEC-NS")] = 5,
	batch_size: Annotated[int, typer.Option(min=1)] = 8192,
	device: Annotated[str, typer.Option()] = "cpu",
	output_device: Annotated[
		str | None, typer.Option(help="Embedding cache device; defaults to the model device")
	] = None,
	output_budget_bytes: Annotated[
		int | None, typer.Option(min=1, help="Declared output-cache budget; required for CUDA output caches")
	] = None,
	threads: Annotated[int, typer.Option(min=1)] = 4,
	add_inverse_edges: Annotated[bool, typer.Option("--add-inverse-edges")] = False,
	phase_file: Annotated[
		Path | None, typer.Option(help="Optional atomic JSON phase receipt for a bounded scheduler")
	] = None,
) -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	torch.set_num_threads(threads)
	recording = RecordedWorkload.load(workload, checkpoint=checkpoint)
	if recording.spec.name != "sage":
		raise typer.BadParameter("The streaming reference currently supports mean GraphSAGE")
	model_device = torch.device(device)
	cache_device = torch.device(output_device) if output_device else model_device
	if model_device.type not in ("cpu", "cuda") or cache_device.type not in ("cpu", "cuda"):
		raise typer.BadParameter("Model and output cache devices must be CPU or CUDA")
	if any(value.type == "cuda" and value.index is None for value in (model_device, cache_device)):
		raise typer.BadParameter("Use an explicit CUDA index, such as cuda:0")
	if cache_device.type == "cuda" and cache_device != model_device:
		raise typer.BadParameter("A CUDA output cache must use the model's CUDA device")
	graph = load_graph(data, add_inverse_edges=add_inverse_edges)
	features = graph.ndata.pop("feat")
	label_values, test_mask = load_test_labels(labels, test_split, graph.num_nodes())
	run(
		graph,
		features,
		recording,
		output,
		fanout=fanout if method is Method.NS else None,
		device=model_device,
		output_device=cache_device,
		batch_size=batch_size,
		output_budget_bytes=output_budget_bytes,
		method=method.value,
		phase_file=phase_file,
		labels=label_values,
		test_mask=test_mask,
	)


if __name__ == "__main__":
	app()
