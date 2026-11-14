"""Load prepared graphs, DGL binaries or OGB CSVs into one inference interface."""

import json
from pathlib import Path

import dgl
import numpy as np
import torch
from ogb.io.read_graph_dgl import read_graph_dgl

from lib.dataset.prepare import SCHEMA


def load_graph(path: Path, *, add_inverse_edges: bool = False):
	path = Path(path)
	manifest_path = path / "manifest.json"
	if path.is_dir() and manifest_path.is_file():
		manifest = json.loads(manifest_path.read_text())
		if manifest.get("schema") != SCHEMA or manifest.get("edge_dtype") != "<i8":
			raise ValueError("Unsupported prepared graph schema")
		nodes, edges = manifest["nodes"], manifest["edges"]
		if type(nodes) is not int or type(edges) is not int or nodes < 1 or edges < 0:
			raise ValueError("Invalid prepared graph dimensions")
		endpoints = []
		for name in ("sources.i64", "targets.i64"):
			if (path / name).stat().st_size != edges * 8:
				raise ValueError("Prepared edge file size does not match its manifest")
			# Copy-on-write protects the prepared dataset from runtime mutations.
			array = np.memmap(path / name, dtype="<i8", mode="c", shape=(edges,)) if edges else np.empty(0, dtype="<i8")
			endpoints.append(torch.from_numpy(array))
		features = np.load(path / "features.npy", mmap_mode="c", allow_pickle=False)
		if features.dtype != np.float32 or features.shape != (nodes, manifest["features"]["width"]):
			raise ValueError("Prepared feature shape or dtype does not match its manifest")
		graph = dgl.graph(tuple(endpoints), num_nodes=nodes)
		graph.ndata["feat"] = torch.from_numpy(features)
	elif path.is_file():
		graphs, _ = dgl.load_graphs(str(path))
		if len(graphs) != 1:
			raise ValueError("An inference dataset must contain exactly one graph")
		graph = graphs[0]
	else:
		graph = read_graph_dgl(str(path))[0]
	if "feat" not in graph.ndata:
		raise ValueError("Inference data must contain node features")
	features = graph.ndata["feat"]
	if features.ndim != 2 or features.shape[1] < 1 or features.device.type != "cpu":
		raise ValueError("Inference requires a nonempty CPU node-feature matrix")
	# CSV readers commonly expose column-major arrays. Normalize once at ingress,
	# not during every sparse row gather. Prepared row-major mappings stay zero-copy.
	graph.ndata["feat"] = features.float().contiguous()
	if add_inverse_edges:
		graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True, exclude_self=False)
	return graph
