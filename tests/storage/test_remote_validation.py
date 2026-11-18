import shutil

import dgl
import pytest

from lib.storage import remote
from lib.storage.base import BaseTopology
from lib.storage.capacity import GIB
from scripts.storage_server import validate_init


def test_fingerprint_matches_exported_content_across_different_paths(tmp_path, monkeypatch):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=5)
	client = BaseTopology.export(graph, tmp_path / "client-base")
	server_folder = tmp_path / "server-base"
	shutil.copytree(client.folder, server_folder)
	server = BaseTopology(server_folder, graph.num_nodes(), graph.num_edges())
	monkeypatch.setattr(remote, "_FINGERPRINT_CHUNK_BYTES", 16)

	assert remote.base_fingerprint(client) == remote.graph_fingerprint(graph)
	assert remote.base_fingerprint(server) == remote.base_fingerprint(client)
	args = [str(client.folder), 5, 3, 0, 5, 1024, 1024, GIB, 0, remote.base_fingerprint(client)]
	validate_init(
		args,
		base=server,
		start=0,
		stop=5,
		capacity_bytes=GIB,
		reserve_bytes=0,
		overlay_bytes=1024,
		response_bytes=1024,
	)
	assert client.folder.exists() and server.folder.exists()


@pytest.mark.parametrize(
	("field", "value", "message"),
	[
		(1, 4, "topology partition"),
		(4, 4, "topology partition"),
		(7, GIB + 1, "node budget"),
	],
)
def test_validate_init_rejects_dimension_partition_and_budget_mismatches(tmp_path, field, value, message):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=5)
	base = BaseTopology.export(graph, tmp_path / "base")
	args = ["different-host-path", 5, 3, 0, 5, 1024, 1024, GIB, 0, remote.base_fingerprint(base)]
	args[field] = value

	with pytest.raises(ValueError, match=message):
		validate_init(
			args,
			base=base,
			start=0,
			stop=5,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)


def test_validate_init_rejects_a_same_dimensions_wrong_graph(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=5)
	wrong = dgl.graph(([0, 1, 3], [1, 3, 2]), num_nodes=5)
	base = BaseTopology.export(graph, tmp_path / "base")
	wrong_base = BaseTopology.export(wrong, tmp_path / "wrong-base")
	args = ["other", 5, 3, 0, 5, 1024, 1024, GIB, 0, remote.base_fingerprint(wrong_base)]

	with pytest.raises(ValueError, match="graph content"):
		validate_init(
			args,
			base=base,
			start=0,
			stop=5,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)
