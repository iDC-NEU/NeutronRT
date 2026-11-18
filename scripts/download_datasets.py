"""Download and register graph datasets on the shared data volume.

Each dataset gets an isolated directory. Existing completed directories are reused;
the downloader never overwrites a prepared dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import dgl
import numpy as np
import torch
import typer

from lib.dataset.download import download
from lib.dataset.offline_ogb import papers100m as load_offline_papers100m
from lib.dataset.prepare import SCHEMA, prepare_edges

DEFAULT_ROOT = Path("/mnt/nfs0/lvhr/neutronrt-evaluation-data/datasets")
FRIENDSTER_URL = "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
REDDIT_URL = "https://snap.stanford.edu/graphsage/reddit.zip"
FRIENDSTER_NODES = 65_608_366
FRIENDSTER_EDGES = 1_806_067_135
REDDIT_CONVERSION_SCHEMA = "neutronrt.reddit.snap.v2"
REDDIT_REPRESENTATION = "snap-graphsage-full-undirected-links-expanded"


@dataclass(frozen=True)
class DatasetRecord:
	name: str
	root: str
	source: str
	status: str
	archive: str | None = None
	archive_bytes: int | None = None
	archive_sha256: str | None = None
	nodes: int | None = None
	edges: int | None = None
	features: int | None = None
	partial: bool | None = None


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


def _write_record(root: Path, record: DatasetRecord) -> None:
	_write_json_atomic(root / "dataset-manifest.json", asdict(record))


def _write_json_atomic(path: Path, value: Any) -> None:
	"""Publish a JSON manifest without exposing a partially-written file."""
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
		) as temporary_stream:
			temporary = Path(temporary_stream.name)
			json.dump(value, temporary_stream, indent=2)
			temporary_stream.write("\n")
			temporary_stream.flush()
			os.fsync(temporary_stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def _sha256_file(path: Path) -> str:
	with path.open("rb") as stream:
		return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
	if not path.is_file():
		raise ValueError(f"{description} is missing: {path}")
	try:
		value = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f"{description} is not valid JSON: {path}") from error
	if not isinstance(value, dict):
		raise ValueError(f"{description} must be a JSON object: {path}")
	return value


def _validate_npy_file(path: Path, *, shape: tuple[int, ...], dtype: np.dtype, description: str) -> None:
	"""Validate both the NPY header and the bytes advertised by that header."""
	if not path.is_file():
		raise ValueError(f"Prepared artifact is incomplete: missing {description} ({path})")
	try:
		array = np.load(path, mmap_mode="r", allow_pickle=False)
		actual_shape = tuple(int(value) for value in array.shape)
		actual_dtype = np.dtype(array.dtype)
		with path.open("rb") as stream:
			version = np.lib.format.read_magic(stream)
			reader = getattr(np.lib.format, f"read_array_header_{version[0]}_{version[1]}", None)
			if reader is None:
				raise ValueError(f"unsupported NPY header version {version}")
			header_shape, _, header_dtype = reader(stream)
			data_offset = stream.tell()
	except (OSError, ValueError, TypeError) as error:
		raise ValueError(f"Prepared {description} has an invalid NPY header or is truncated: {path}") from error

	if actual_shape != shape or tuple(header_shape) != shape:
		raise ValueError(f"Prepared {description} shape mismatch: expected {shape}, found {actual_shape}")
	if actual_dtype != dtype or np.dtype(header_dtype) != dtype:
		raise ValueError(f"Prepared {description} dtype mismatch: expected {dtype}, found {actual_dtype}")
	expected_bytes = data_offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
	actual_bytes = path.stat().st_size
	if actual_bytes != expected_bytes:
		raise ValueError(f"Prepared {description} size mismatch: expected {expected_bytes} bytes, found {actual_bytes}")


def _validate_friendster_prepared(
	prepared: Path,
	*,
	source_sha256: str,
	nodes: int,
	expected_edges: int | None,
	feature_dim: int,
	seed: int,
) -> dict[str, Any]:
	"""Validate a completed Friendster preparation before reusing it."""
	manifest = _read_json(prepared / "manifest.json", "Friendster prepared manifest")
	problems: list[str] = []

	def require(name: str, expected: Any) -> None:
		if manifest.get(name) != expected:
			problems.append(f"{name}={manifest.get(name)!r}, expected {expected!r}")

	require("schema", SCHEMA)
	if manifest.get("partial") is not False:
		problems.append(f"partial={manifest.get('partial')!r}, expected False")
	if type(manifest.get("nodes")) is not int:
		problems.append(f"nodes={manifest.get('nodes')!r} is not an integer")
	require("nodes", nodes)
	if type(manifest.get("expected_edges")) not in (int, type(None)) or isinstance(
		manifest.get("expected_edges"), bool
	):
		problems.append(f"expected_edges={manifest.get('expected_edges')!r} is not an integer or null")
	require("expected_edges", expected_edges)
	require("node_id_policy", "sorted-compact")
	require("original_ids", "original_ids.i64")
	if manifest.get("source_undirected") is not True:
		problems.append(f"source_undirected={manifest.get('source_undirected')!r}, expected True")
	require("directed", True)
	require("edge_dtype", "<i8")
	require("labels", None)
	if str(manifest.get("source_sha256", "")).lower() != source_sha256.lower():
		problems.append("source_sha256 does not match the downloaded archive")

	input_edges = manifest.get("input_edges")
	if type(input_edges) is not int or input_edges < 1:
		problems.append(f"input_edges={input_edges!r} is not a positive integer")
	elif expected_edges is not None and input_edges != expected_edges:
		problems.append(f"input_edges={input_edges!r}, expected {expected_edges!r}")

	self_loops = manifest.get("self_loops")
	edges = manifest.get("edges")
	if type(self_loops) is not int or type(edges) is not int:
		problems.append("edges and self_loops must be integer counts")
	else:
		if self_loops < 0 or isinstance(input_edges, int) and self_loops > input_edges:
			problems.append(f"self_loops={self_loops!r} is inconsistent with input_edges={input_edges!r}")
		expected_arcs = input_edges + (input_edges - self_loops) if isinstance(input_edges, int) else None
		if expected_arcs is not None and edges != expected_arcs:
			problems.append(
				f"edges={edges!r} is inconsistent with {input_edges!r} input edges and {self_loops!r} self loops"
			)

	features = manifest.get("features")
	if not isinstance(features, dict):
		problems.append("features metadata is missing")
	else:
		if type(features.get("width")) is not int or type(features.get("seed")) is not int:
			problems.append("features.width and features.seed must be integers")
		for name, expected in (
			("kind", "synthetic-normal"),
			("width", feature_dim),
			("dtype", "float32"),
			("seed", seed),
		):
			if features.get(name) != expected:
				problems.append(f"features.{name}={features.get(name)!r}, expected {expected!r}")

	if problems:
		raise ValueError("Incompatible Friendster prepared artifact; refusing to overwrite it: " + "; ".join(problems))

	assert isinstance(edges, int)
	_validate_npy_file(
		prepared / "features.npy",
		shape=(nodes, feature_dim),
		dtype=np.dtype("<f4"),
		description="feature matrix",
	)
	for name in ("sources.i64", "targets.i64"):
		path = prepared / name
		if not path.is_file():
			raise ValueError(f"Prepared artifact is incomplete: missing {name} ({path})")
		actual = path.stat().st_size
		expected = edges * np.dtype("<i8").itemsize
		if actual != expected:
			raise ValueError(f"Prepared {name} size mismatch: expected {expected} bytes, found {actual}")
	original_ids = prepared / "original_ids.i64"
	if not original_ids.is_file():
		raise ValueError(f"Prepared artifact is incomplete: missing original_ids.i64 ({original_ids})")
	actual = original_ids.stat().st_size
	expected = nodes * np.dtype("<i8").itemsize
	if actual != expected:
		raise ValueError(f"Prepared original_ids.i64 size mismatch: expected {expected} bytes, found {actual}")
	return manifest


def _validate_reddit_graph(graph, *, metadata: dict[str, Any]) -> dict[str, int]:
	"""Validate the converted graph tensors and return its dimensions."""
	nodes = int(graph.num_nodes())
	edges = int(graph.num_edges())
	if type(metadata.get("nodes")) is not int or type(metadata.get("edges")) is not int:
		raise ValueError("Reddit conversion metadata must record integer graph dimensions")
	if nodes < 1 or edges < 1 or nodes != metadata["nodes"] or edges != metadata["edges"]:
		raise ValueError(
			"Reddit conversion graph dimensions do not match metadata: "
			f"graph=({nodes}, {edges}), metadata=({metadata['nodes']}, {metadata['edges']})"
		)
	feature_width = metadata.get("feature_width")
	if type(feature_width) is not int or feature_width < 1:
		raise ValueError("Reddit conversion metadata must record a positive feature width")
	try:
		features = graph.ndata["feat"]
		labels = graph.ndata["label"]
		masks = [graph.ndata[f"{name}_mask"] for name in ("train", "val", "test")]
	except KeyError as error:
		raise ValueError(f"Reddit conversion graph is missing required node data: {error.args[0]}") from error
	if (
		features.dtype != torch.float32
		or features.ndim != 2
		or tuple(features.shape)
		!= (
			nodes,
			feature_width,
		)
	):
		raise ValueError("Reddit conversion graph feature dtype or shape is invalid")
	if labels.dtype != torch.int64 or labels.ndim != 1 or tuple(labels.shape) != (nodes,):
		raise ValueError("Reddit conversion graph labels must be int64 with one value per node")
	if labels.numel() and labels.min().item() < 0:
		raise ValueError("Reddit conversion graph labels must be nonnegative class IDs")
	for name, mask in zip(("train", "val", "test"), masks, strict=True):
		if mask.dtype not in (torch.bool, torch.uint8) or mask.ndim != 1 or tuple(mask.shape) != (nodes,):
			raise ValueError(f"Reddit conversion graph {name} split mask is invalid")
		if mask.dtype == torch.uint8 and mask.numel() and mask.max().item() > 1:
			raise ValueError(f"Reddit conversion graph {name} split mask contains non-binary values")
	if torch.stack(masks).to(torch.int8).sum(dim=0).ne(1).any():
		raise ValueError("Reddit conversion graph train/val/test masks must partition the nodes")
	return {"nodes": nodes, "edges": edges, "feature_width": int(features.shape[1])}


def _validate_reddit_conversion_metadata(metadata: dict[str, Any], *, archive_sha256: str, archive_bytes: int) -> None:
	"""Check archive identity, source members, and the exact conversion policy."""
	expected = {
		"schema": REDDIT_CONVERSION_SCHEMA,
		"archive_sha256": archive_sha256,
		"archive_bytes": archive_bytes,
		"representation": REDDIT_REPRESENTATION,
		"source": "SNAP GraphSAGE full topology",
		"topology_member": "reddit-G_full.json",
		"features_member": "reddit-feats.npy",
		"labels_member": "reddit-class_map.json",
		"splits_member": "reddit-G_full.json",
		"feature_dtype": "float32",
		"label_dtype": "int64",
		"split_names": ["train", "val", "test"],
		"self_loop_reverse_excluded": True,
	}
	problems = [
		f"{key}={metadata.get(key)!r}, expected {value!r}"
		for key, value in expected.items()
		if metadata.get(key) != value
	]
	if metadata.get("endpoint_encoding") not in ("positions", "labels"):
		problems.append("endpoint_encoding must identify positions or labels")
	source_directed = metadata.get("source_directed")
	if type(source_directed) is not bool:
		problems.append("source_directed must be a boolean")
	elif metadata.get("expand_undirected_links") != (not source_directed):
		problems.append(
			f"expand_undirected_links={metadata.get('expand_undirected_links')!r} is inconsistent with "
			f"source_directed={source_directed!r}"
		)
	if problems:
		raise ValueError(
			"Incompatible Reddit conversion metadata; refusing to relabel the existing graph: " + "; ".join(problems)
		)


def _validate_reddit_graph_file(graph_path: Path, metadata: dict[str, Any]) -> None:
	"""Verify the persisted DGL graph is the exact file published with metadata."""
	if not graph_path.is_file():
		raise ValueError(f"Reddit conversion graph is missing: {graph_path}")
	if type(metadata.get("graph_bytes")) is not int or metadata["graph_bytes"] != graph_path.stat().st_size:
		raise ValueError("Reddit conversion graph size does not match metadata; refusing to reuse it")
	graph_sha256 = metadata.get("graph_sha256")
	if not isinstance(graph_sha256, str) or _sha256_file(graph_path) != graph_sha256.lower():
		raise ValueError("Reddit conversion graph hash does not match metadata; refusing to reuse it")


def _reddit_endpoint_rows(topology: dict, ids: dict) -> tuple[list[int], str]:
	"""Resolve old positional node-link endpoints without confusing feature rows."""
	node_ids = [str(node["id"]) for node in topology["nodes"]]
	if len(node_ids) != len(ids) or len(set(node_ids)) != len(ids) or set(node_ids) != set(ids):
		raise ValueError("SNAP topology nodes and feature ID map disagree")
	rows = [ids[name] for name in node_ids]
	encoding = None
	for edge in topology["links"]:
		for endpoint in (edge["source"], edge["target"]):
			positional = type(endpoint) is int and 0 <= endpoint < len(rows)
			label = str(endpoint) in ids
			if not positional and not label:
				raise ValueError(f"SNAP edge endpoint cannot be resolved: {endpoint!r}")
			if positional != label:
				encoding = "positions" if positional else "labels"
				break
		if encoding:
			break
	if encoding is None:
		# If every possible numeric label and position maps to the same row,
		# either interpretation is equivalent. Otherwise require an explicit
		# format rather than silently choosing the wrong vertex identities.
		if any(str(index) in ids and ids[str(index)] != row for index, row in enumerate(rows)):
			raise ValueError("Ambiguous SNAP node-link endpoint encoding")
		encoding = "labels"
	return rows, encoding


def _reddit(root: Path, *, url: str, connections: int) -> DatasetRecord:
	path = root / "reddit"
	graph_path = path / "dgl_graph.bin"
	conversion_path = path / "conversion-manifest.json"
	receipt = download(url, root / "reddit.zip", connections=connections)
	archive_sha256 = str(receipt["sha256"])
	archive_bytes = int(receipt["bytes"]) if "bytes" in receipt else (root / "reddit.zip").stat().st_size
	if graph_path.is_file():
		metadata = _read_json(conversion_path, "Reddit conversion metadata")
		_validate_reddit_conversion_metadata(metadata, archive_sha256=archive_sha256, archive_bytes=archive_bytes)
		_validate_reddit_graph_file(graph_path, metadata)
		graph = dgl.load_graphs(str(graph_path))[0][0]
		_validate_reddit_graph(graph, metadata=metadata)
	elif conversion_path.exists():
		raise ValueError("Reddit conversion metadata exists but dgl_graph.bin is missing; refusing to rebuild in place")
	else:
		# SNAP GraphSAGE archives are not DGL RedditDataset archives. Read the
		# supplied full topology directly; never fall back to data.dgl.ai.
		with zipfile.ZipFile(root / "reddit.zip") as stream:

			def member(suffix):
				matches = [name for name in stream.namelist() if name.endswith(suffix)]
				if len(matches) != 1:
					raise ValueError(f"Expected one SNAP archive member ending in {suffix}")
				return stream.open(matches[0])

			with member("reddit-id_map.json") as source:
				ids = json.load(source)
			with member("reddit-class_map.json") as source:
				classes = json.load(source)
			with member("reddit-G_full.json") as source:
				topology = json.load(source)
			with member("reddit-feats.npy") as source:
				features = np.load(source, allow_pickle=False)
		if sorted(ids.values()) != list(range(len(ids))) or len(features) != len(ids):
			raise ValueError("SNAP node mapping and features disagree")
		links = topology["links"]
		position_rows, endpoint_encoding = _reddit_endpoint_rows(topology, ids)

		def endpoint_row(value):
			if endpoint_encoding == "positions":
				if type(value) is not int or not 0 <= value < len(position_rows):
					raise ValueError("SNAP positional edge endpoint is outside topology nodes")
				return position_rows[value]
			return ids[str(value)]

		u = torch.tensor([endpoint_row(edge["source"]) for edge in links], dtype=torch.int64)
		v = torch.tensor([endpoint_row(edge["target"]) for edge in links], dtype=torch.int64)
		if not topology.get("directed", False):
			# Expand undirected links once, excluding duplicate reverse self loops.
			mask = u != v
			u, v = torch.cat((u, v[mask])), torch.cat((v, u[mask]))
		graph = dgl.graph((u, v), num_nodes=len(ids))
		graph.ndata["feat"] = torch.from_numpy(features.astype(np.float32))
		labels = torch.empty(len(ids), dtype=torch.int64)
		for key, row in ids.items():
			value = classes[key]
			labels[row] = int(np.argmax(value)) if isinstance(value, list) else int(value)
		graph.ndata["label"] = labels
		for name in ("train", "val", "test"):
			graph.ndata[name + "_mask"] = torch.zeros(len(ids), dtype=torch.bool)
		for node in topology["nodes"]:
			name = "test" if node.get("test") else "val" if node.get("val") else "train"
			graph.ndata[name + "_mask"][ids[str(node["id"])]] = True
		_validate_reddit_graph(
			graph,
			metadata={
				"nodes": int(graph.num_nodes()),
				"edges": int(graph.num_edges()),
				"feature_width": int(graph.ndata["feat"].shape[1]),
			},
		)
		path.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile(dir=path, prefix=".dgl-", suffix=".tmp", delete=False) as temporary_stream:
			temporary = Path(temporary_stream.name)
		try:
			dgl.save_graphs(str(temporary), [graph])
			# Publish without replacing a graph that another preparation just
			# completed. The temporary file is on the same filesystem.
			os.link(temporary, graph_path)
		finally:
			temporary.unlink(missing_ok=True)
		metadata = {
			"schema": REDDIT_CONVERSION_SCHEMA,
			"source": "SNAP GraphSAGE full topology",
			"archive": "reddit.zip",
			"archive_sha256": archive_sha256,
			"archive_bytes": archive_bytes,
			"topology_member": "reddit-G_full.json",
			"features_member": "reddit-feats.npy",
			"labels_member": "reddit-class_map.json",
			"splits_member": "reddit-G_full.json",
			"source_directed": bool(topology.get("directed", False)),
			"representation": REDDIT_REPRESENTATION,
			"endpoint_encoding": endpoint_encoding,
			"expand_undirected_links": not bool(topology.get("directed", False)),
			"self_loop_reverse_excluded": True,
			"nodes": int(graph.num_nodes()),
			"edges": int(graph.num_edges()),
			"graph_bytes": int(graph_path.stat().st_size),
			"graph_sha256": _sha256_file(graph_path),
			"feature_width": int(graph.ndata["feat"].shape[1]),
			"feature_dtype": "float32",
			"label_dtype": "int64",
			"split_names": ["train", "val", "test"],
		}
		_validate_reddit_graph(graph, metadata=metadata)
		_validate_reddit_conversion_metadata(metadata, archive_sha256=archive_sha256, archive_bytes=archive_bytes)
		_write_json_atomic(conversion_path, metadata)
	record = DatasetRecord(
		name="reddit",
		root=str(path),
		source="SNAP GraphSAGE full topology; raw features; undirected links expanded",
		status="ready",
		archive=str(root / "reddit.zip"),
		archive_bytes=int((root / "reddit.zip").stat().st_size) if (root / "reddit.zip").is_file() else None,
		archive_sha256=archive_sha256,
		nodes=int(graph.num_nodes()),
		edges=int(graph.num_edges()),
		features=int(graph.ndata["feat"].shape[1]),
	)
	_write_record(root, record)
	return record


def _friendster(
	root: Path,
	*,
	url: str,
	nodes: int,
	expected_edges: int | None,
	feature_dim: int,
	connections: int,
	chunk_edges: int,
	seed: int,
) -> DatasetRecord:
	raw = root / "raw" / "edges.txt.gz"
	prepared = root / "prepared"
	receipt = download(url, raw, connections=connections)
	if prepared.exists() and not prepared.is_dir():
		raise ValueError(f"Friendster prepared path is not a directory; refusing to overwrite it: {prepared}")
	if not prepared.exists():
		manifest = prepare_edges(
			raw,
			prepared,
			nodes=nodes,
			feature_dim=feature_dim,
			seed=seed,
			chunk_edges=chunk_edges,
			expected_edges=expected_edges,
			source_sha256=receipt["sha256"],
			compact_ids=True,
			undirected=True,
		)
	manifest = _validate_friendster_prepared(
		prepared,
		source_sha256=str(receipt["sha256"]),
		nodes=nodes,
		expected_edges=expected_edges,
		feature_dim=feature_dim,
		seed=seed,
	)
	record = DatasetRecord(
		name="friendster",
		root=str(prepared),
		source=url,
		status="ready" if not manifest.get("partial") else "partial",
		archive=str(raw),
		archive_bytes=int(receipt["bytes"]),
		archive_sha256=str(receipt["sha256"]),
		nodes=int(manifest["nodes"]),
		edges=int(manifest["edges"]),
		features=int(manifest["features"]["width"]),
		partial=bool(manifest["partial"]),
	)
	_write_record(root, record)
	return record


def _papers100m(root: Path) -> DatasetRecord:
	dataset = load_offline_papers100m(root)
	graph = dataset[0][0]
	path = Path(dataset.root)
	record = DatasetRecord(
		name="papers100M",
		root=str(path / "processed" / "dgl_data_processed"),
		source="OGB ogbn-papers100M",
		status="ready",
		nodes=int(graph.num_nodes()),
		edges=int(graph.num_edges()),
		features=int(graph.ndata["feat"].shape[1]),
	)
	_write_record(root, record)
	return record


@app.command("reddit")
def reddit(
	root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
	url: Annotated[str, typer.Option("--url")] = REDDIT_URL,
	connections: Annotated[int, typer.Option("--connections")] = 8,
):
	"""Download DGL Reddit and write its manifest."""
	root.mkdir(parents=True, exist_ok=True)
	typer.echo(json.dumps(asdict(_reddit(root, url=url, connections=connections)), indent=2))


@app.command("friendster")
def friendster(
	root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
	url: Annotated[str, typer.Option("--url")] = FRIENDSTER_URL,
	nodes: Annotated[int, typer.Option("--nodes")] = FRIENDSTER_NODES,
	expected_edges: Annotated[int | None, typer.Option("--expected-edges")] = FRIENDSTER_EDGES,
	feature_dim: Annotated[int, typer.Option("--feature-dim")] = 128,
	connections: Annotated[int, typer.Option("--connections")] = 8,
	chunk_edges: Annotated[int, typer.Option("--chunk-edges")] = 1_000_000,
	seed: Annotated[int, typer.Option("--seed")] = 71,
):
	"""Download Friendster and prepare bounded-memory synthetic features."""
	root.mkdir(parents=True, exist_ok=True)
	record = _friendster(
		root,
		url=url,
		nodes=nodes,
		expected_edges=expected_edges,
		feature_dim=feature_dim,
		connections=connections,
		chunk_edges=chunk_edges,
		seed=seed,
	)
	typer.echo(json.dumps(asdict(record), indent=2))


@app.command("papers100m")
def papers100m(
	root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
):
	"""Download and preprocess OGBN-Papers100M with the official OGB reader."""
	root.mkdir(parents=True, exist_ok=True)
	typer.echo(json.dumps(asdict(_papers100m(root)), indent=2))


@app.command("all")
def all_datasets(
	root: Annotated[Path, typer.Option("--root")] = DEFAULT_ROOT,
	reddit_url: Annotated[str, typer.Option("--reddit-url")] = REDDIT_URL,
	reddit_connections: Annotated[int, typer.Option("--reddit-connections")] = 8,
	friendster_connections: Annotated[int, typer.Option("--friendster-connections")] = 8,
):
	"""Download Reddit, Friendster, and Papers100M sequentially."""
	root.mkdir(parents=True, exist_ok=True)
	records = [
		_reddit(root, url=reddit_url, connections=reddit_connections),
		_friendster(
			root,
			url=FRIENDSTER_URL,
			nodes=FRIENDSTER_NODES,
			expected_edges=FRIENDSTER_EDGES,
			feature_dim=128,
			connections=friendster_connections,
			chunk_edges=1_000_000,
			seed=71,
		),
		_papers100m(root),
	]
	_write_record(
		root,
		DatasetRecord(name="datasets", root=str(root), source="official dataset sources", status="ready"),
	)
	typer.echo(json.dumps([asdict(record) for record in records], indent=2))


if __name__ == "__main__":
	app()
