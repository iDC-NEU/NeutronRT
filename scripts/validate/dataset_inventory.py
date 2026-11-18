"""Inspect the fixed dataset registry without loading graphs or downloading data."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer


def _json(path: Path) -> dict:
	return dict(json.loads(path.read_text()))


def _file(path: Path, expected_hash: str | None, full: bool) -> dict:
	result = {"path": str(path), "bytes": path.stat().st_size}
	if full:
		with path.open("rb") as stream:
			digest = hashlib.file_digest(stream, "sha256").hexdigest()
		result["sha256"] = digest
		if expected_hash is not None and digest != expected_hash:
			raise ValueError(f"Checksum mismatch: {path}")
	return result


def inspect(entry: dict, *, full: bool = False) -> dict:
	"""Metadata inspection is lightweight; full checksums require neu-sbox."""
	root = Path(entry["path"])
	result = {**entry, "status": "pending", "verification": "full-checksum" if full else "metadata-only"}
	if not root.exists():
		result["reason"] = "prepared dataset is not present"
		return result
	files = []
	try:
		if entry["format"] == "prepared":
			manifest = _json(root / "manifest.json")
			if manifest.get("schema") != "neutronrt.graph.v1" or manifest.get("partial") is not False:
				raise ValueError("Prepared graph must have the complete supported schema")
			if manifest.get("edge_dtype") != "<i8":
				raise ValueError("Prepared edge dtype differs")
			nodes, edges = manifest["nodes"], manifest["edges"]
			for name in ("sources.i64", "targets.i64"):
				if (root / name).stat().st_size != edges * 8:
					raise ValueError(f"Edge array size differs: {name}")
				files.append(_file(root / name, manifest.get("edge_sha256", {}).get(name), full))
			features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
			if features.shape != (nodes, manifest["features"]["width"]) or features.dtype != np.float32:
				raise ValueError("Feature header differs from the manifest")
			expected_bytes = features.offset + features.nbytes
			del features
			if (root / "features.npy").stat().st_size != expected_bytes:
				raise ValueError("Feature file size differs from its header")
			files.append(_file(root / "features.npy", None, full))
			result.update(nodes=nodes, directed_edges=edges, manifest=manifest)
		elif entry["format"] == "reddit-dgl":
			manifest = _json(root / "conversion-manifest.json")
			if manifest.get("schema") != "neutronrt.reddit.snap.v2":
				raise ValueError("Reddit conversion is stale or unsupported")
			graph = root / "dgl_graph.bin"
			if graph.stat().st_size != manifest["graph_bytes"]:
				raise ValueError("Reddit graph size differs")
			files.append(_file(graph, manifest["graph_sha256"], full))
			result.update(nodes=manifest["nodes"], directed_edges=manifest["edges"], manifest=manifest)
		elif entry["format"] == "ogb-csv":
			for key, filename in (("nodes", "num-node-list.csv.gz"), ("stored_edges", "num-edge-list.csv.gz")):
				with gzip.open(root / filename, "rt") as stream:
					result[key] = int(stream.readline().strip())
			for name in ("edge.csv.gz", "node-feat.csv.gz", "node-label.csv.gz"):
				files.append(_file(root / name, None, full))
		elif entry["format"] == "ogb-papers":
			processed = root / "processed/dgl_data_processed"
			if not processed.is_file():
				result["reason"] = "OGB processing is not complete"
				return result
			files.append(_file(processed, None, full))
		else:
			raise ValueError("Unknown dataset format")
		result.update(status="present", files=files)
	except (OSError, ValueError, KeyError, TypeError) as error:
		result.update(status="invalid", reason=f"{type(error).__name__}: {error}")
	return result


def main(
	registry: Annotated[Path, typer.Option(exists=True)] = Path("config/experiments/datasets.json"),
	output: Annotated[Path, typer.Option()] = Path("logs/dataset-inventory.json"),
	full: Annotated[bool, typer.Option(help="Read all bytes; run only through neu-sbox")] = False,
	prepare_papers: Annotated[
		bool, typer.Option(help="Prepare Papers100M offline from its completed ZIP, then require all six datasets")
	] = False,
):
	if output.exists():
		raise typer.BadParameter("Output exists; choose a new receipt path")
	entries = _json(registry)["datasets"]
	if prepare_papers:
		papers = next(entry for entry in entries if entry["format"] == "ogb-papers")
		archive = Path(papers["archive"])
		if not archive.is_file():
			raise typer.BadParameter("Completed Papers100M ZIP is required; preparation will not download it")
		from scripts.download_datasets import _papers100m

		_papers100m(archive.parent)
	records = [inspect(entry, full=full) for entry in entries]
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x") as stream:
		json.dump({"registry": str(registry), "records": records}, stream, indent=2)
		stream.write("\n")
	for record in records:
		typer.echo(f"{record['tier']:5} {record['name']:12} {record['status']} ({record['verification']})")
	if any(record["status"] == "invalid" or prepare_papers and record["status"] != "present" for record in records):
		raise typer.Exit(1)


if __name__ == "__main__":
	typer.run(main)
