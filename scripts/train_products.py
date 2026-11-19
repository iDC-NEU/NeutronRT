"""Train the shared Products GraphSAGE checkpoint on the prepared replay graph."""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import dgl
import pandas as pd
import torch
import torch.nn.functional as F
import typer

from benchmarks.dgl import layerwise_infer
from lib.dataset.inference import load_graph
from lib.models import GNNModel
from scripts.train_baseline import accuracy


def main(
	graph_path: Path,
	dataset: Path,
	output: Path,
	epochs: Annotated[int, typer.Option(min=1)] = 20,
	device: str = "cuda:0",
):
	output.mkdir(parents=True, exist_ok=False)
	torch.set_num_threads(4)
	torch.manual_seed(42)
	dgl.seed(42)
	graph = load_graph(graph_path)
	features = graph.ndata["feat"]
	labels_path = dataset / "raw/node-label.csv.gz"
	split_paths = {name: dataset / f"split/sales_ranking/{name}.csv.gz" for name in ("train", "valid", "test")}
	labels = torch.from_numpy(pd.read_csv(labels_path, header=None).to_numpy(copy=True).reshape(-1)).long()
	splits = {
		name: torch.from_numpy(pd.read_csv(path, header=None).to_numpy(copy=True).reshape(-1)).long()
		for name, path in split_paths.items()
	}
	nodes = graph.num_nodes()
	if labels.shape != (nodes,) or ((labels < 0) | (labels >= 47)).any():
		raise ValueError("Products labels do not match the prepared graph")
	seen = torch.zeros(nodes, dtype=torch.bool)
	for ids in splits.values():
		if not ids.numel() or ids.min() < 0 or ids.max() >= nodes or ids.unique().numel() != ids.numel():
			raise ValueError("Invalid Products split IDs")
		if seen[ids].any():
			raise ValueError("Products splits overlap")
		seen[ids] = True

	config = {
		"model_name": "sage",
		"num_features": features.shape[1],
		"num_hiddens": 64,
		"num_classes": 47,
		"num_layers": 2,
		"aggregator_type": "mean",
	}
	model = GNNModel(**config).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	loader = dgl.dataloading.DataLoader(
		graph,
		splits["train"],
		dgl.dataloading.NeighborSampler([10, 10]),
		batch_size=1024,
		shuffle=True,
		drop_last=False,
		num_workers=0,
	)
	best_score = -1.0
	history = []
	for epoch in range(1, epochs + 1):
		model.train()
		epoch_loss = torch.zeros((), device=device)
		for input_ids, output_ids, blocks in loader:
			optimizer.zero_grad(set_to_none=True)
			values = features[input_ids].to(device)
			for layer, block in enumerate(blocks):
				values = model(layer, block.to(device), values)
			loss = F.cross_entropy(values, labels[output_ids].to(device))
			loss.backward()
			optimizer.step()
			epoch_loss += loss.detach() * output_ids.numel()
		model.eval()
		logits, _ = layerwise_infer(graph, features, model, batch_size=256)
		valid = accuracy(logits, labels, {"valid": splits["valid"]})["valid"]
		del logits
		record = {"epoch": epoch, "loss": float(epoch_loss.cpu()) / splits["train"].numel(), "valid": valid}
		history.append(record)
		print(json.dumps(record), flush=True)
		if valid > best_score:
			best_score = valid
			best_epoch = epoch
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

	model.load_state_dict(best_state)
	logits, _ = layerwise_infer(graph, features, model, batch_size=256)
	scores = accuracy(logits, labels, splits)
	checkpoint = output / "checkpoint.pt"
	torch.save(
		{
			"model_config": config,
			"state_dict": best_state,
			"dataset": "ogbn-products",
			"seed": 42,
			"epoch": best_epoch,
			"scores": scores,
		},
		checkpoint,
	)
	report = {
		"graph": str(graph_path),
		"nodes": nodes,
		"edges": graph.num_edges(),
		"input_sha256": {
			str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (labels_path, *split_paths.values())
		},
		"split_sizes": {name: ids.numel() for name, ids in splits.items()},
		"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
		"model_config": config,
		"best_epoch": best_epoch,
		"scores": scores,
		"history": history,
	}
	(output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	typer.run(main)
