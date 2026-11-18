"""Export persistent native-DGL CSR/CSC bases once for reuse by experiment cohorts."""

import gc
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Annotated

import dgl
import torch
import typer

from lib.dataset.inference import load_graph
from lib.storage.base import BaseTopology
from lib.storage.remote import base_fingerprint


def export(entry: dict, destination: Path) -> dict:
	"""Publish a new immutable base; never overwrite a dataset or existing export."""
	if destination.exists():
		raise FileExistsError(f"Topology destination already exists: {destination}")
	path = Path(entry["path"])
	if entry["format"] == "reddit-dgl":
		path = path / "dgl_graph.bin"
	elif entry["format"] == "ogb-papers":
		path = path / "processed/dgl_data_processed"
	started = perf_counter()
	logging.info("Loading %s from %s", entry["name"], path)
	graph = load_graph(path, add_inverse_edges=entry.get("add_inverse_edges", False))
	loaded = perf_counter()
	destination.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix=f".{entry['name']}-", dir=destination.parent) as temporary:
		staging = Path(temporary)
		base = BaseTopology.export(graph, staging / "base")
		exported = perf_counter()
		fingerprint = base_fingerprint(base)
		# Small-tier OGB CSVs otherwise need parsing on every launch. Preserve the
		# native graph/feature representation using DGL's own binary serializer.
		inference_path = str(path)
		if entry["format"] == "ogb-csv":
			dgl.save_graphs(str(staging / "graph.bin"), [graph], formats=["coo"])
			inference_path = str(destination / "graph.bin")
		files = {p.name: p.stat().st_size for p in sorted((staging / "base").iterdir()) if p.is_file()}
		result = {
			"schema": "neutronrt.topology-export.v1",
			"dataset": entry["name"],
			"tier": entry["tier"],
			"source": str(path),
			"inference_path": inference_path,
			"add_inverse_edges": entry.get("add_inverse_edges", False),
			"inference_graph_already_has_inverse_edges": entry.get("add_inverse_edges", False),
			"topology_base": str(destination / "base"),
			"nodes": base.nodes,
			"edges": base.edges,
			"features": {"kind": entry["features"], "width": graph.ndata["feat"].shape[1]},
			"labels_available_in_source": entry["labels"],
			"base_fingerprint": fingerprint,
			"base_files_bytes": files,
			"dgl_version": dgl.__version__,
			"timing": {
				"load_seconds": loaded - started,
				"export_seconds": exported - loaded,
				"validation_and_serialization_seconds": perf_counter() - exported,
			},
			"storage": "int64 NumPy CSR/CSC files; immutable mapped base, bounded overlays at runtime",
			"provenance": "standard DGL adjacency conversion; preserves multiedges, self loops and declared direction",
		}
		(staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
		os.rename(staging, destination)
	del graph
	gc.collect()
	return result


def main(
	registry: Annotated[Path, typer.Option(exists=True)] = Path("config/experiments/datasets.json"),
	output: Annotated[Path, typer.Option()] = Path("logs/topology-exports"),
	datasets: Annotated[list[str] | None, typer.Option("--dataset")] = None,
	threads: Annotated[int, typer.Option(min=1)] = 8,
):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	torch.set_num_threads(threads)
	settings = json.loads(registry.read_text())
	entries = settings["datasets"]
	if datasets:
		unknown = set(datasets) - {entry["name"] for entry in entries}
		if unknown:
			raise typer.BadParameter(f"Unknown datasets: {sorted(unknown)}")
		entries = [entry for entry in entries if entry["name"] in datasets]
	output.mkdir(parents=True, exist_ok=False)
	results = []
	for entry in entries:
		result = export(entry, output / entry["name"])
		results.append(result)
		(output / "summary.json").write_text(
			json.dumps(
				{
					"registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
					"datasets": results,
				},
				indent=2,
			)
			+ "\n"
		)
		logging.info("Published %s: nodes=%d edges=%d", entry["name"], result["nodes"], result["edges"])


if __name__ == "__main__":
	typer.run(main)
