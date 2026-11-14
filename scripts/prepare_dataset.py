"""Download topology and prepare a bounded-memory inference dataset."""

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import typer

from lib.dataset.download import download, file_sha256
from lib.dataset.prepare import prepare_edges

TWITTER_URL = "https://snap.stanford.edu/data/twitter-2010.txt.gz"
# Complete archive record count; SNAP's summary table lists 298 fewer edges.
# Preserve the archive's records, including self loops, rather than filtering it.
TWITTER_EDGES = 1468365182


DatasetChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"DatasetChoice", {str(value).replace("-", "_").upper(): value for value in ["twitter", "edge-list"]}
)


@dataclass
class Options:
	dataset: str
	root: Path
	input: Path | None
	url: str | None
	sha256: str | None
	connections: int
	nodes: int | None
	expected_edges: int | None
	feature_dim: int
	seed: int
	chunk_edges: int
	limit_edges: int | None


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	dataset: Annotated[DatasetChoice, typer.Argument()],
	root: Annotated[Path, typer.Option("--root", help="Local data directory, excluded from source exports")],
	input: Annotated[Path | None, typer.Option("--input", help="Existing edge list; bypass download")] = None,
	url: Annotated[
		str | None, typer.Option("--url", help="Public edge-list archive URL; defaults to SNAP Twitter topology")
	] = None,
	sha256: Annotated[
		str | None, typer.Option("--sha256", help="Expected archive SHA-256 when supplied by a trusted source")
	] = None,
	connections: Annotated[int, typer.Option("--connections", help="1–16 resumable HTTP range transfers")] = 8,
	nodes: Annotated[
		int | None, typer.Option("--nodes", help="Declared node count; preserves isolated vertices")
	] = None,
	expected_edges: Annotated[
		int | None, typer.Option("--expected-edges", help="Reject a complete import with a different count")
	] = None,
	feature_dim: Annotated[int, typer.Option("--feature-dim")] = 128,
	seed: Annotated[int, typer.Option("--seed")] = 71,
	chunk_edges: Annotated[int, typer.Option("--chunk-edges")] = 1000000,
	limit_edges: Annotated[
		int | None, typer.Option("--limit-edges", help="Explicit partial import; never reported as the full dataset")
	] = None,
):
	args = Options(**{**locals(), "dataset": dataset.value})
	if args.dataset == "edge-list" and args.input is None and args.url is None:
		raise typer.BadParameter("edge-list requires --input or --url")
	if min(args.feature_dim, args.chunk_edges) < 1 or args.nodes is not None and args.nodes < 1:
		raise typer.BadParameter("Feature, node and chunk sizes must be positive")
	if args.limit_edges is not None and args.limit_edges < 1:
		raise typer.BadParameter("Edge limit must be positive")
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	args.root.mkdir(parents=True, exist_ok=True)
	source = args.input
	receipt = None
	if source is None:
		source = (
			args.root
			/ "raw"
			/ ("edges.txt.gz" if urlsplit(args.url or TWITTER_URL).path.endswith(".gz") else "edges.txt")
		)
		receipt = download(args.url or TWITTER_URL, source, sha256=args.sha256, connections=args.connections)
		logging.info("Archive complete: %d bytes; SHA-256 %s", receipt["bytes"], receipt["sha256"])
	elif args.sha256 is not None and file_sha256(source) != args.sha256.lower():
		raise ValueError("Input archive checksum does not match")
	destination = args.root / "prepared"
	manifest = prepare_edges(
		source,
		destination,
		nodes=args.nodes if args.nodes is not None else 41652230 if args.dataset == "twitter" else None,
		feature_dim=args.feature_dim,
		seed=args.seed,
		chunk_edges=args.chunk_edges,
		expected_edges=(
			args.expected_edges
			if args.expected_edges is not None
			else TWITTER_EDGES
			if args.dataset == "twitter"
			else None
		),
		limit_edges=args.limit_edges,
		source_sha256=receipt["sha256"] if receipt else None,
	)
	print(json.dumps({"nodes": manifest["nodes"], "edges": manifest["edges"], "partial": manifest["partial"]}))


if __name__ == "__main__":
	app()
