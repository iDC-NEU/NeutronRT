"""Exercise loss of unflushed GPU embedding state on ogbn-arxiv."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Annotated, Optional, Sequence

import torch
import typer
from dgl import DGLGraph
from torch import Tensor, nn

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.data_plane import LocalEmbeddingPlane, RejectedPublication, copy_tensor
from lib.graph.ownership import MissingState
from lib.graph.serving import EmbeddingService
from scripts.validate.recompute import apply_layer, build_model


@torch.inference_mode()
def run_case(
	graph: DGLGraph,
	features: Tensor,
	base: nn.ModuleList,
	query: Tensor,
	expected: Tensor,
	devices: Sequence[str],
	*,
	before_writeback: bool,
) -> dict:
	started, proceed = Event(), Event()
	if not before_writeback:
		proceed.set()

	def transfer(value, device):
		if value.is_cuda and device.type == "cpu":
			started.set()
			if not proceed.wait(60):
				raise TimeoutError("Validation writeback gate timed out")
		return copy_tensor(value, device)

	plane = LocalEmbeddingPlane(transfer=transfer)
	try:
		storage = plane.add_node(0, "cpu", 64 * 1024**2)
		workers = [plane.add_node(index + 1, device, 64 * 1024**2) for index, device in enumerate(devices)]
		models = {node.owner: copy.deepcopy(base).to(node.device) for node in workers}

		def apply(worker, task, inputs):
			if not inputs.is_cuda:
				raise AssertionError("GNN computation must execute on the GPU")
			return apply_layer(models[worker.owner], task.layer - 1, task.block.to(inputs.device), inputs)

		widths = [features.shape[1], *[int(layer._out_feats) for layer in base]]
		service = EmbeddingService(graph, features, widths, plane, workers, apply)
		service.ensure(query)
		if storage.store.bytes_used:
			raise AssertionError("Completion unexpectedly required master materialization")
		key = service.catalog.selections(len(base), query, 0)[0].key
		producer = plane.directory.locate(key).owner
		future = plane.writeback(key, storage.owner)
		if not started.wait(30):
			raise TimeoutError("Writeback did not start")
		if not before_writeback:
			future.result(timeout=30)
		before = service.counters.tasks
		lost = service.recover(producer)
		if before_writeback:
			if not lost or future.done() or storage.store.bytes_used:
				raise AssertionError("Recovery must finish while old writeback remains pending")
			proceed.set()
			try:
				future.result(timeout=30)
			except RejectedPublication:
				pass
			else:
				raise AssertionError("Late writeback from a lost owner was accepted")
			try:
				plane.directory.locate(key)
			except MissingState:
				pass
			else:
				raise AssertionError("Obsolete state remained advertised after recovery")
		elif lost or service.counters.tasks != before:
			raise AssertionError("A surviving storage replica should avoid recomputation")
		with service.query(query, storage.owner) as actual:
			torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
			error = float((actual - expected).abs().max())
			disagreement = float((actual.argmax(1) != expected.argmax(1)).float().mean())
		return {
			"failure_timing": "before writeback acknowledgment"
			if before_writeback
			else "after writeback acknowledgment",
			"query_rows": query.numel(),
			"producer_rank": producer.rank,
			"lost_states": len(lost),
			"recompute_tasks": service.counters.tasks - before,
			"recovery_rows": service.counters.recovery_rows,
			"recovery_seconds": service.counters.recovery_seconds,
			"max_abs_error": error,
			"prediction_disagreement": disagreement,
		}
	finally:
		proceed.set()
		plane.close()


@torch.inference_mode()
def validate(raw: Path, checkpoint: Optional[Path], devices: Sequence[str], query_size: int) -> dict:
	graph = load_graph_data(Path(raw))
	features = graph.ndata.pop("feat")
	graph.create_formats_()
	generator = torch.Generator().manual_seed(71)
	query = torch.randint(graph.num_nodes(), (query_size,), generator=generator)
	records = []
	for name in ("sage", "gat"):
		base = build_model(name, features.shape[1], checkpoint)
		model = copy.deepcopy(base).to(devices[0])
		full_graph = graph.to(devices[0])
		full = features.to(devices[0])
		for layer in range(len(model)):
			full = apply_layer(model, layer, full_graph, full)
		expected = full[query.to(devices[0])].cpu()
		for before in (True, False):
			record = run_case(graph, features, base, query, expected, devices, before_writeback=before)
			record["model"] = name
			records.append(record)
			print(json.dumps(record), flush=True)
	return {
		"dataset": "ogbn-arxiv",
		"devices": devices,
		"checkpoint": str(checkpoint) if checkpoint else None,
		"scope": (
			"In-process endpoint-state loss with real GPU operators and delayed writeback; "
			"OS process failure, network failure, and NCCL group recovery are not tested"
		),
		"records": records,
	}


@dataclass
class Options:
	data: Path
	checkpoint: Path | None
	devices: tuple[str, str]
	query_size: int
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	devices: Annotated[tuple[str, str], typer.Option("--devices")] = ("cuda:0", "cuda:1"),
	query_size: Annotated[int, typer.Option("--query-size")] = 128,
):
	args = Options(**locals())
	if args.query_size < 1 or len(set(args.devices)) != 2:
		raise typer.BadParameter("Need a positive query size and two distinct compute devices")
	torch.set_num_threads(4)
	result = validate(args.data, args.checkpoint, args.devices, args.query_size)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
	app()
