"""Download and register the large revision datasets on the shared data volume.

Each dataset gets an isolated directory. Existing completed directories are reused;
the downloader never overwrites a prepared dataset.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
from dgl.data.reddit import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

from lib.dataset.download import download
from lib.dataset.prepare import prepare_edges

DEFAULT_ROOT = Path("/mnt/nfs0/lvhr/neutronrt-evaluation-data/revision-datasets")
FRIENDSTER_URL = "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
REDDIT_URL = "https://snap.stanford.edu/graphsage/reddit.zip"
FRIENDSTER_NODES = 65_608_366
FRIENDSTER_EDGES = 1_806_067_135


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
	(root / "dataset-manifest.json").write_text(json.dumps(asdict(record), indent=2) + "\n")


def _reddit(root: Path, *, url: str, connections: int) -> DatasetRecord:
	path = root / "reddit"
	graph_path = path / "dgl_graph.bin"
	receipt = None
	if not graph_path.is_file():
		archive = root / "reddit.zip"
		receipt = download(url, archive, connections=connections)
		if not path.exists():
			path.mkdir(parents=True)
			with zipfile.ZipFile(archive) as stream:
				stream.extractall(path)
		RedditDataset(raw_dir=str(root), force_reload=False, verbose=True)
	graph = RedditDataset(raw_dir=str(root), force_reload=False, verbose=False)[0]
	record = DatasetRecord(
		name="reddit",
		root=str(path),
		source="DGL RedditDataset",
		status="ready",
		archive=str(root / "reddit.zip"),
		archive_bytes=int((root / "reddit.zip").stat().st_size) if (root / "reddit.zip").is_file() else None,
		archive_sha256=receipt["sha256"] if receipt is not None else None,
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
	if not prepared.is_dir():
		manifest = prepare_edges(
			raw,
			prepared,
			nodes=nodes,
			feature_dim=feature_dim,
			seed=seed,
			chunk_edges=chunk_edges,
			expected_edges=expected_edges,
			source_sha256=receipt["sha256"],
		)
	else:
		manifest = json.loads((prepared / "manifest.json").read_text())
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
	dataset = DglNodePropPredDataset(name="ogbn-papers100M", root=str(root))
	graph = dataset[0][0]
	path = root / "ogbn_papers100M"
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
		DatasetRecord(name="revision", root=str(root), source="official dataset sources", status="ready"),
	)
	typer.echo(json.dumps([asdict(record) for record in records], indent=2))


if __name__ == "__main__":
	app()
