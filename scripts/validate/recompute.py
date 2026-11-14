"""Validate sparse recomputation against full GPU inference on raw ogbn-arxiv."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Dict, Optional, Tuple, cast

import dgl.nn as dglnn
import torch
import typer
from dgl import DGLGraph
from torch import Tensor, nn

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.recompute import LayerPlan, RecomputePlanner, execute_recomputation


def build_model(name: str, features: int, checkpoint: Optional[Path]) -> nn.ModuleList:
	torch.manual_seed(42)
	data = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint is not None else None
	config = data["model_config"] if data is not None else {}
	if config and (config["model_name"] != "sage" or config["num_features"] != features):
		raise ValueError("Expected an arXiv GraphSAGE baseline checkpoint")
	depth = int(config.get("num_layers", 2))
	hidden = int(config.get("num_hiddens", 64))
	classes = int(config.get("num_classes", 40))
	model = nn.ModuleList()
	for layer in range(depth):
		inputs = features if layer == 0 else hidden
		outputs = classes if layer + 1 == depth else hidden
		if name == "sage":
			model.append(dglnn.SAGEConv(inputs, outputs, "mean"))
		else:
			model.append(dglnn.GATConv(inputs, outputs, 2, residual=True, allow_zero_in_degree=True))
	if name == "sage" and data is not None:
		model.load_state_dict({key.removeprefix("layers."): value for key, value in data["state_dict"].items()})
	return model.eval()


def apply_layer(model: nn.ModuleList, layer: int, graph: DGLGraph, features: Tensor) -> Tensor:
	output = model[layer](graph, features)
	if output.ndim == 3:
		output = output.mean(1)
	return cast(Tensor, output if layer + 1 == len(model) else output.relu())


@torch.inference_mode()
def sparse_query(graph: DGLGraph, features: Tensor, model: nn.ModuleList, query: Tensor, version: int, device: str):
	states: Dict[int, Tuple[Tensor, Tensor]] = {}
	plan = RecomputePlanner(graph, len(model)).plan(query, len(model), version)

	def load(layer: int, nodes: Tensor, requested: int) -> Tensor:
		if requested != version:
			raise ValueError("Graph version mismatch")
		if layer == 0:
			return features.index_select(0, nodes).to(device)
		positions, values = states[layer]
		indices = positions[nodes]
		if (indices < 0).any():
			raise ValueError("Input dependency was not produced")
		return values.index_select(0, indices.to(device))

	def apply(task: LayerPlan, inputs: Tensor) -> Tensor:
		return apply_layer(model, task.layer - 1, task.block.to(device), inputs)

	def publish(layer: int, nodes: Tensor, values: Tensor, requested: int) -> None:
		if requested != version:
			raise ValueError("Publication version mismatch")
		positions = torch.full((graph.num_nodes(),), -1, dtype=torch.int64)
		positions.index_copy_(0, nodes, torch.arange(nodes.numel()))
		states[layer] = positions, values

	execute_recomputation(plan, load=load, apply=apply, publish=publish)
	positions, values = states[len(model)]
	return values.index_select(0, positions[query].to(device)), plan


@torch.inference_mode()
def validate(raw: Path, checkpoint: Optional[Path], device: str, query_size: int) -> dict:
	graph = load_graph_data(Path(raw))
	features = graph.ndata.pop("feat")
	generator = torch.Generator().manual_seed(71)
	query = torch.randint(graph.num_nodes(), (query_size,), generator=generator)
	records = []
	for name in ("sage", "gat"):
		current = graph.clone()
		model = build_model(name, features.shape[1], checkpoint).to(device)
		generator.manual_seed(71)
		changed = query.unique()[:16]
		previous = features[changed].clone()
		previous_query = None
		try:
			for version, operation in enumerate(("initialize", "features", "insert", "delete")):
				if operation == "features":
					features.index_copy_(0, changed, previous + torch.randn(previous.shape, generator=generator) * 0.05)
				elif operation == "insert":
					u = torch.randint(graph.num_nodes(), (16,), generator=generator)
					v = query[torch.randint(query.numel(), (16,), generator=generator)]
					current.add_edges(u, v)
				elif operation == "delete":
					incoming = current.in_edges(query.unique(), form="eid")
					eids = incoming[torch.randperm(incoming.numel(), generator=generator)[:16]]
					current.remove_edges(eids)
				full_graph = current.to(device)
				full = features.to(device)
				for layer in range(len(model)):
					full = apply_layer(model, layer, full_graph, full)
				start = perf_counter()
				actual, plan = sparse_query(current, features, model, query, version, device)
				expected = full[query.to(device)]
				torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
				changed_rows = (
					int(((expected - previous_query).abs().amax(1) > 2e-5).sum()) if previous_query is not None else 0
				)
				if version and not changed_rows:
					raise AssertionError("Update trace did not change any queried embeddings")
				previous_query = expected.clone()
				record = {
					"model": name,
					"weights": "trained baseline" if name == "sage" and checkpoint else "fixed random weights",
					"version": version,
					"operation": operation,
					"query_rows": query.numel(),
					"max_abs_error": float((actual - expected).abs().max()),
					"prediction_disagreement": float((actual.argmax(1) != expected.argmax(1)).float().mean()),
					"changed_query_rows": changed_rows,
					"check_seconds": perf_counter() - start,
					"layer_destinations": [task.targets.numel() for task in plan.tasks],
					"layer_edges": [task.block.num_edges() for task in plan.tasks],
				}
				records.append(record)
				print(json.dumps(record), flush=True)
		finally:
			features.index_copy_(0, changed, previous)
	return {
		"dataset": "ogbn-arxiv",
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"device": device,
		"checkpoint": str(checkpoint) if checkpoint else None,
		"scope": (
			"Sparse GPU recomputation correctness; these timings do not measure deployed throughput or accuracy loss"
		),
		"records": records,
	}


@dataclass
class Options:
	data: Path
	checkpoint: Path | None
	device: str
	query_size: int
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	device: Annotated[str, typer.Option("--device")] = "cuda:0",
	query_size: Annotated[int, typer.Option("--query-size")] = 128,
):
	args = Options(**locals())
	if args.query_size < 1:
		raise typer.BadParameter("query-size must be positive")
	torch.set_num_threads(4)
	result = validate(args.data, args.checkpoint, args.device, args.query_size)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
	app()
