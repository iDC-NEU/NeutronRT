import json

import dgl
import pytest
import torch

from lib.storage.base import BaseTopology
from lib.storage.remote import base_fingerprint, graph_fingerprint
from scripts.prepare_topology import export


def test_export_preserves_parallel_edges_and_reusable_native_binary(tmp_path, monkeypatch):
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 2]), num_nodes=4)
	graph.ndata["feat"] = torch.arange(12, dtype=torch.float32).reshape(4, 3)
	monkeypatch.setattr("scripts.prepare_topology.load_graph", lambda *args, **kwargs: graph)
	entry = {
		"name": "fixture",
		"tier": "small",
		"format": "ogb-csv",
		"path": str(tmp_path),
		"features": "real",
		"labels": True,
	}
	out = tmp_path / "export"
	result = export(entry, out)
	reloaded = dgl.load_graphs(str(out / "graph.bin"))[0][0]
	assert reloaded.num_edges() == 4
	torch.testing.assert_close(reloaded.ndata["feat"], graph.ndata["feat"])
	base = BaseTopology(out / "base", graph.num_nodes(), graph.num_edges())
	assert base_fingerprint(base) == graph_fingerprint(reloaded) == result["base_fingerprint"]
	assert json.loads((out / "manifest.json").read_text())["edges"] == 4
	with pytest.raises(FileExistsError):
		export(entry, out)


def test_failed_export_does_not_publish_partial_base(tmp_path, monkeypatch):
	graph = dgl.graph(([0], [1]), num_nodes=2)
	graph.ndata["feat"] = torch.ones(2, 3)
	monkeypatch.setattr("scripts.prepare_topology.load_graph", lambda *args, **kwargs: graph)

	def fail(*args, **kwargs):
		raise OSError("fixture interrupted write")

	monkeypatch.setattr("scripts.prepare_topology.base_fingerprint", fail)
	entry = {
		"name": "fixture",
		"tier": "large",
		"format": "prepared",
		"path": str(tmp_path),
		"features": "synthetic-normal",
		"labels": False,
	}
	with pytest.raises(OSError, match="interrupted"):
		export(entry, tmp_path / "export")
	assert not (tmp_path / "export").exists()
