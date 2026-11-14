"""Prepare large edge lists without float parsing, global sorting or account maps."""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lib.dataset.download import file_sha256

SCHEMA = "neutronrt.graph.v1"


def prepare_edges(
	source: Path,
	destination: Path,
	*,
	nodes: int | None = None,
	feature_dim: int = 128,
	seed: int = 71,
	chunk_edges: int = 1000000,
	expected_edges: int | None = None,
	limit_edges: int | None = None,
	source_sha256: str | None = None,
) -> dict:
	"""Preserve directed edges, duplicates, self loops and numeric source IDs.

	Only one input chunk and a bounded feature chunk are resident. Features are
	explicitly synthetic: an unlabelled topology is not a classification dataset.
	The completed directory is published only after gzip CRC/parsing and counts
	pass. A limited import is recorded as partial and cannot claim full validation.
	"""
	source, destination = Path(source), Path(destination)
	if destination.exists():
		raise FileExistsError("Prepared destination exists; choose a new directory")
	if feature_dim < 1 or chunk_edges < 1 or nodes is not None and nodes < 1:
		raise ValueError("Node, feature and chunk dimensions must be positive")
	if limit_edges is not None and limit_edges < 1:
		raise ValueError("Edge limit must be positive")
	destination.parent.mkdir(parents=True, exist_ok=True)
	digest = source_sha256 or file_sha256(source)
	with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
		folder = Path(temporary)
		count, maximum, self_loops = 0, -1, 0
		edge_hashes = [hashlib.sha256(), hashlib.sha256()]
		with (folder / "sources.i64").open("wb") as sources, (folder / "targets.i64").open("wb") as targets:
			reader = pd.read_csv(source, sep=r"\s+", header=None, comment="#", dtype=np.int64, chunksize=chunk_edges)
			with reader:
				for frame in reader:
					if frame.shape[1] != 2:
						raise ValueError("An edge list must have exactly two integer columns")
					if limit_edges is not None:
						frame = frame.iloc[: limit_edges - count]
					values = frame.to_numpy(copy=False)
					if values.size:
						if values.min() < 0 or nodes is not None and values.max() >= nodes:
							raise ValueError("Edge node ID is outside the declared graph")
						maximum = max(maximum, int(values.max()))
						self_loops += int(np.count_nonzero(values[:, 0] == values[:, 1]))
					for index, stream in enumerate((sources, targets)):
						column = np.ascontiguousarray(values[:, index], dtype="<i8")
						stream.write(memoryview(column).cast("B"))
						edge_hashes[index].update(memoryview(column).cast("B"))
					count += len(frame)
					logging.info("Prepared %d edges", count)
					if limit_edges is not None and count >= limit_edges:
						break
		if not count:
			raise ValueError("The edge list contains no edges")
		if limit_edges is None and expected_edges is not None and count != expected_edges:
			raise ValueError(f"Expected {expected_edges} edges; parsed {count}")
		node_count = nodes if nodes is not None else maximum + 1
		features = np.lib.format.open_memmap(
			folder / "features.npy", mode="w+", dtype=np.float32, shape=(node_count, feature_dim)
		)
		generator = np.random.default_rng(seed)
		feature_rows = max(1, (64 * 1024**2) // (4 * feature_dim))
		for start in range(0, node_count, feature_rows):
			stop = min(start + feature_rows, node_count)
			features[start:stop] = generator.standard_normal((stop - start, feature_dim), dtype=np.float32) * 0.1
			logging.info("Prepared features for %d / %d vertices", stop, node_count)
		features.flush()
		del features
		manifest = {
			"schema": SCHEMA,
			"nodes": node_count,
			"edges": count,
			"self_loops": self_loops,
			"directed": True,
			"edge_dtype": "<i8",
			"source_sha256": digest,
			"edge_sha256": dict(zip(("sources.i64", "targets.i64"), (h.hexdigest() for h in edge_hashes), strict=True)),
			"features": {"kind": "synthetic-normal", "width": feature_dim, "dtype": "float32", "seed": seed},
			"labels": None,
			"partial": limit_edges is not None,
			"expected_edges": expected_edges,
		}
		(folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
		os.rename(folder, destination)
	return manifest
