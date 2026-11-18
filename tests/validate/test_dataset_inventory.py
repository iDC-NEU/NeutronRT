import hashlib
import json

import numpy as np
import typer
from typer.testing import CliRunner

from scripts.validate.dataset_inventory import inspect, main


def prepared(tmp_path):
	np.array([0, 1], dtype="<i8").tofile(tmp_path / "sources.i64")
	np.array([1, 0], dtype="<i8").tofile(tmp_path / "targets.i64")
	np.save(tmp_path / "features.npy", np.ones((2, 3), dtype=np.float32))
	manifest = {
		"schema": "neutronrt.graph.v1",
		"partial": False,
		"nodes": 2,
		"edges": 2,
		"edge_dtype": "<i8",
		"features": {"width": 3},
		"edge_sha256": {
			name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in ("sources.i64", "targets.i64")
		},
	}
	(tmp_path / "manifest.json").write_text(json.dumps(manifest))
	return {"name": "fixture", "tier": "large", "format": "prepared", "path": str(tmp_path)}


def test_missing_papers_remains_pending_without_creating_or_downloading(tmp_path):
	path = tmp_path / "missing"
	result = inspect({"name": "papers", "format": "ogb-papers", "path": str(path)})
	assert result["status"] == "pending"
	assert not path.exists()


def test_metadata_detects_truncated_feature_payload(tmp_path):
	entry = prepared(tmp_path)
	assert inspect(entry)["status"] == "present"
	path = tmp_path / "features.npy"
	path.write_bytes(path.read_bytes()[:-4])
	assert inspect(entry)["status"] == "invalid"


def test_full_audit_detects_same_size_edge_corruption(tmp_path):
	entry = prepared(tmp_path)
	assert inspect(entry, full=True)["status"] == "present"
	path = tmp_path / "sources.i64"
	path.write_bytes(bytes(path.stat().st_size))
	assert inspect(entry)["status"] == "present"  # Metadata is deliberately weaker.
	result = inspect(entry, full=True)
	assert result["status"] == "invalid" and "Checksum mismatch" in result["reason"]


def test_preparation_requires_completed_archive_without_starting_reader(tmp_path, monkeypatch):
	registry = tmp_path / "registry.json"
	registry.write_text(
		json.dumps(
			{
				"datasets": [
					{
						"name": "papers",
						"format": "ogb-papers",
						"path": str(tmp_path / "prepared"),
						"archive": str(tmp_path / "papers.zip"),
					}
				]
			}
		)
	)
	app = typer.Typer()
	app.command()(main)
	result = CliRunner().invoke(
		app, ["--registry", str(registry), "--output", str(tmp_path / "report.json"), "--prepare-papers"]
	)
	assert result.exit_code != 0
	assert "Completed Papers100M ZIP is required" in result.output
	assert not (tmp_path / "prepared").exists()
