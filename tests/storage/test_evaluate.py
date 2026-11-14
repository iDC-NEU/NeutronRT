import json

import dgl
import pytest
import torch
from typer.testing import CliRunner

from scripts.evaluate.storage import app

pytestmark = pytest.mark.host


def test_storage_probe_reports_read_work_and_round_trip_mutations_without_compute(tmp_path, monkeypatch):
	graph = dgl.graph(([0, 0, 1, 2, 8, 11], [0, 1, 2, 4, 9, 12]), num_nodes=16)
	graph.ndata["feat"] = torch.ones(16, 4)
	monkeypatch.setattr("scripts.evaluate.storage.load_graph_data", lambda *args, **kwargs: graph)
	path = tmp_path / "probe.json"
	result = CliRunner().invoke(
		app,
		[
			"--root",
			str(tmp_path),
			"--output",
			str(path),
			"--nodes",
			"8",
			"--trials",
			"2",
			"--partitions",
			"1",
			"--partitions",
			"2",
		],
	)
	assert result.exit_code == 0, result.output

	report = json.loads(path.read_text())
	assert len(report["dgl_reads"]) == 2
	assert [case["partitions"] for case in report["settings"]] == [1, 2]
	for case in report["settings"]:
		assert case["vs_dgl_read_ratio"] > 0 and case["vs_one_partition_ratio"] > 0
		assert len(case["trials"]) == 2
		assert all(row["insert_seconds"] > 0 and row["delete_seconds"] > 0 for row in case["trials"])
		assert case["profile"]["initialization"]["base_file_bytes"] > 0
	assert not torch.cuda.is_initialized()
