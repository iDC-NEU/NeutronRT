"""Read official extracted OGB archives without another processed-data copy."""

from pathlib import Path

import pandas as pd
import torch
from ogb.io.read_graph_dgl import read_graph_dgl
from ogb.nodeproppred import dataset_dgl


class RawOGBDataset:
	def __init__(self, name, root):
		meta = pd.read_csv(Path(dataset_dgl.__file__).with_name("master.csv"), index_col=0, keep_default_na=False)[name]
		if meta["is hetero"] == "True" or meta["binary"] == "True":
			raise ValueError("Raw loading currently supports homogeneous CSV datasets")
		self.directory = Path(root) / name.replace("-", "_")
		if not (self.directory / "raw" / "edge.csv.gz").is_file():
			self.directory = Path(root) / meta["download_name"]
		self.num_classes = int(meta["num classes"])
		self.split = meta["split"]

		def additional(key):
			return [] if meta[key] == "None" else meta[key].split(",")

		self.graph = read_graph_dgl(
			str(self.directory / "raw"),
			add_inverse_edge=meta["add_inverse_edge"] == "True",
			additional_node_files=additional("additional node files"),
			additional_edge_files=additional("additional edge files"),
		)[0]
		labels = pd.read_csv(self.directory / "raw" / "node-label.csv.gz", header=None).to_numpy(copy=True)
		self.graph.ndata["label"] = torch.from_numpy(labels).squeeze(-1).nan_to_num(-1).long()

	def __getitem__(self, index):
		if index != 0:
			raise IndexError(index)
		return self.graph

	def __len__(self):
		return 1

	def get_idx_split(self):
		return {
			name: torch.from_numpy(
				pd.read_csv(self.directory / "split" / self.split / f"{name}.csv.gz", header=None)
				.to_numpy(copy=True)
				.reshape(-1)
			).long()
			for name in ("train", "valid", "test")
		}


def has_raw_ogb(name, root):
	return any(
		(Path(root) / directory / "raw" / "edge.csv.gz").is_file()
		for directory in (name.replace("-", "_"), name.removeprefix("ogbn-"))
	)
