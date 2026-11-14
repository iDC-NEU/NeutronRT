"""Train a small full-neighbor GraphSAGE baseline on the official OGB split."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

import dgl
import torch
import torch.nn.functional as F
import typer

from lib.dataset.raw_ogb import RawOGBDataset
from lib.models import GNNModel


def predict(model, graph, features):
	values = features
	for layer in range(model.num_layers):
		values = model(layer, graph, values)
	return values


@torch.inference_mode()
def accuracy(logits, labels, splits):
	predictions = logits.argmax(dim=1)
	return {name: float((predictions[nodes] == labels[nodes]).float().mean()) for name, nodes in splits.items()}


@dataclass
class Options:
	root: str
	output: Path
	report: Path
	device: str
	epochs: int
	hidden: int
	seed: int


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	root: Annotated[str, typer.Option("--root")],
	output: Annotated[Path, typer.Option("--output")],
	report: Annotated[Path, typer.Option("--report")],
	device: Annotated[str, typer.Option("--device")] = "cuda:0",
	epochs: Annotated[int, typer.Option("--epochs")] = 30,
	hidden: Annotated[int, typer.Option("--hidden")] = 64,
	seed: Annotated[int, typer.Option("--seed")] = 42,
):
	args = Options(**locals())
	_run(args)


def _run(args: Options) -> None:
	if args.epochs < 1 or args.hidden < 1:
		raise typer.BadParameter("Epochs and hidden size must be positive")
	torch.manual_seed(args.seed)
	dgl.seed(args.seed)
	data = RawOGBDataset("ogbn-arxiv", args.root)
	graph = data[0].to(args.device)
	features = graph.ndata["feat"]
	labels = graph.ndata["label"]
	splits = {name: nodes.to(args.device) for name, nodes in data.get_idx_split().items()}
	config = {
		"model_name": "sage",
		"num_features": features.shape[1],
		"num_hiddens": args.hidden,
		"num_classes": data.num_classes,
		"num_layers": 2,
		"aggregator_type": "mean",
	}
	model = GNNModel(**config).to(args.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	history = []
	best = None
	start = perf_counter()
	for epoch in range(1, args.epochs + 1):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		logits = predict(model, graph, features)
		loss = F.cross_entropy(logits[splits["train"]], labels[splits["train"]])
		loss.backward()
		optimizer.step()
		model.eval()
		with torch.inference_mode():
			scores = accuracy(predict(model, graph, features), labels, splits)
		record = {"epoch": epoch, "loss": float(loss.detach()), **scores}
		history.append(record)
		if best is None or scores["valid"] > best["scores"]["valid"]:
			best = {
				"model_config": config,
				"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
				"dataset": "ogbn-arxiv",
				"seed": args.seed,
				"epoch": epoch,
				"scores": scores,
			}
		print(json.dumps(record), flush=True)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	assert best is not None
	torch.save(best, args.output)
	report = {
		"dataset": "ogbn-arxiv",
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"split_sizes": {key: value.numel() for key, value in splits.items()},
		"model_config": config,
		"seed": args.seed,
		"epochs": args.epochs,
		"checkpoint": str(args.output),
		"best_epoch": best["epoch"],
		"scores": best["scores"],
		"history": history,
		"seconds": perf_counter() - start,
		"scope": "Small baseline on the original directed graph; no claim of leaderboard accuracy",
	}
	args.report.parent.mkdir(parents=True, exist_ok=True)
	args.report.write_text(json.dumps(report, indent=2) + "\n")
	print(json.dumps({"checkpoint": str(args.output), "scores": best["scores"]}), flush=True)


if __name__ == "__main__":
	app()
