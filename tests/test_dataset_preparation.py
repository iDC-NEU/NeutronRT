import gzip
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import numpy as np
import pytest
import torch

from lib.dataset.download import download
from lib.dataset.inference import load_graph
from lib.dataset.prepare import prepare_edges


def test_csv_ingress_normalizes_column_major_features_once(monkeypatch, tmp_path):
	import dgl

	graph = dgl.graph(([0], [1]), num_nodes=3)
	array = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
	graph.ndata["feat"] = torch.from_numpy(array)
	assert not graph.ndata["feat"].is_contiguous()
	monkeypatch.setattr("lib.dataset.inference.read_graph_dgl", lambda path: [graph])
	loaded = load_graph(tmp_path / "raw")
	assert loaded.ndata["feat"].is_contiguous()
	torch.testing.assert_close(loaded.ndata["feat"], torch.from_numpy(array))


@contextmanager
def _server(payload, *, ignore_range=False, truncate_once=False, head_supported=True):
	state, lock = {"truncated": False}, Lock()

	class Handler(BaseHTTPRequestHandler):
		def do_HEAD(self):
			if not head_supported:
				self.send_error(405)
				return
			self.send_response(200)
			self.send_header("ETag", '"fixed-graph"')
			self.send_header("Accept-Ranges", "bytes")
			self.send_header("Content-Length", str(len(payload)))
			self.end_headers()

		def do_GET(self):
			begin, end = self.headers.get("Range", "bytes=0-")[6:].split("-")
			offset = int(begin) if not ignore_range else 0
			stop = int(end) + 1 if end and not ignore_range else len(payload)
			self.send_response(206 if self.headers.get("Range") and not ignore_range else 200)
			self.send_header("ETag", '"fixed-graph"')
			self.send_header("Content-Length", str(stop - offset))
			if self.headers.get("Range") and not ignore_range:
				self.send_header("Content-Range", f"bytes {offset}-{stop - 1}/{len(payload)}")
			self.end_headers()
			with lock:
				truncate = truncate_once and not state["truncated"]
				state["truncated"] |= truncate
			body = payload[offset:stop]
			self.wfile.write(body[: len(body) // 2] if truncate else body)

		def log_message(self, *args):
			pass

	server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
	thread = Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		yield f"http://127.0.0.1:{server.server_port}/edges.txt.gz"
	finally:
		server.shutdown()
		server.server_close()
		thread.join()


@pytest.mark.host
def test_download_falls_back_when_the_server_does_not_implement_head(tmp_path):
	payload = b"0 1\n" * 100
	with _server(payload, head_supported=False) as url:
		target = tmp_path / "edges.txt"
		receipt = download(url, target, connections=8, segment_bytes=64)
	assert target.read_bytes() == payload and receipt["bytes"] == len(payload)


@pytest.mark.host
@pytest.mark.parametrize("ignore_range", [False, True])
def test_download_resumes_or_restarts_without_publishing_a_partial_file(tmp_path, ignore_range):
	payload = gzip.compress(b"0 1\n1 2\n" * 1000)
	target = tmp_path / "edges.txt.gz"
	with _server(payload, ignore_range=ignore_range) as url:
		(target.with_name(target.name + ".part")).write_bytes(payload[:20])
		(target.with_name(target.name + ".part.json")).write_text(
			json.dumps({"url_id": hashlib.sha256(url.encode()).hexdigest(), "validator": '"fixed-graph"'})
		)
		receipt = download(url, target, sha256=hashlib.sha256(payload).hexdigest())
	assert target.read_bytes() == payload
	assert receipt["resumed_bytes"] == (0 if ignore_range else 20)
	assert not list(tmp_path.glob("*.part*"))
	with pytest.raises(ValueError, match="checksum"):
		download(url, target, sha256="0" * 64)
	assert target.read_bytes() == payload


@pytest.mark.host
def test_parallel_ranges_assemble_the_archive_and_resume_existing_segments(tmp_path):
	payload = bytes(range(256)) * 20
	target = tmp_path / "archive"
	with _server(payload) as url:
		folder = target.with_name(target.name + ".segments")
		folder.mkdir()
		(folder / "transfer.json").write_text(
			json.dumps(
				{
					"url_id": hashlib.sha256(url.encode()).hexdigest(),
					"bytes": len(payload),
					"validator": '"fixed-graph"',
					"segment_bytes": 512,
				}
			)
		)
		(folder / "000000").write_bytes(payload[:512])
		(folder / "000001").write_bytes(payload[512:612])
		receipt = download(url, target, connections=3, segment_bytes=512, sha256=hashlib.sha256(payload).hexdigest())
	assert target.read_bytes() == payload and receipt["connections"] == 3
	assert receipt["resumed_bytes"] == 612
	assert not target.with_name(target.name + ".segments").exists()


@pytest.mark.host
def test_parallel_download_retries_truncated_responses_without_losing_cached_bytes(tmp_path):
	payload = bytes(range(256)) * 20
	target = tmp_path / "archive"
	with _server(payload, truncate_once=True) as url:
		receipt = download(url, target, connections=2, segment_bytes=512, sha256=hashlib.sha256(payload).hexdigest())
	assert target.read_bytes() == payload
	assert receipt["range_retries"] == 1


def _prepare(tmp_path, **options):
	source = tmp_path / "edges.txt.gz"
	with gzip.open(source, "wt") as stream:
		stream.write("# topology only\n0 1\n0 1\n2 2\n")
	destination = tmp_path / "prepared"
	manifest = prepare_edges(source, destination, nodes=4, feature_dim=3, chunk_edges=1, **options)
	return source, destination, manifest


def test_streaming_preparation_preserves_edges_and_copy_on_write_features(tmp_path):
	_, destination, manifest = _prepare(tmp_path, expected_edges=3)
	assert manifest["nodes"] == 4 and manifest["edges"] == 3
	assert manifest["self_loops"] == 1
	assert manifest["features"]["kind"] == "synthetic-normal" and manifest["labels"] is None
	assert not manifest["partial"]
	before = np.load(destination / "features.npy").copy()
	graph = load_graph(destination)
	u, v = graph.edges()
	assert u.tolist() == [0, 0, 2] and v.tolist() == [1, 1, 2]
	graph.ndata["feat"][0] += 9
	graph.add_edges(torch.tensor([3]), torch.tensor([0]))
	np.testing.assert_equal(np.load(destination / "features.npy"), before)
	assert load_graph(destination).num_edges() == 3
	assert load_graph(destination, add_inverse_edges=True).num_edges() == 6


def test_failed_or_partial_preparation_cannot_claim_full_dataset_validation(tmp_path):
	source = tmp_path / "edges.txt"
	source.write_text("0 1\n1 3\n")
	with pytest.raises(ValueError, match="outside"):
		prepare_edges(source, tmp_path / "bad", nodes=3, feature_dim=2, chunk_edges=1)
	assert not (tmp_path / "bad").exists()
	assert not list(tmp_path.glob(".prepare-*"))
	with pytest.raises(ValueError, match="Expected 3"):
		prepare_edges(source, tmp_path / "wrong-count", nodes=4, expected_edges=3)
	manifest = prepare_edges(source, tmp_path / "partial", nodes=4, feature_dim=2, limit_edges=1, expected_edges=3)
	assert manifest["partial"] and manifest["edges"] == 1
	assert manifest["expected_edges"] == 3


def test_loader_rejects_truncated_prepared_edges(tmp_path):
	_, destination, _ = _prepare(tmp_path)
	(destination / "sources.i64").write_bytes(b"bad")
	with pytest.raises(ValueError, match="size"):
		load_graph(destination)


def test_downloader_and_preparer_do_not_import_torch_or_dgl():
	subprocess.run(
		[
			sys.executable,
			"-c",
			"import sys, lib.dataset.download, lib.dataset.prepare; "
			"assert 'torch' not in sys.modules and 'dgl' not in sys.modules",
		],
		check=True,
	)


def test_compact_preparation_preserves_sparse_id_identity_and_multiplicity(tmp_path):
	source = tmp_path / "sparse.txt"
	source.write_text("100 7\n100 7\n900 900\n")
	destination = tmp_path / "compact"
	manifest = prepare_edges(source, destination, nodes=3, feature_dim=2, chunk_edges=2, compact_ids=True)
	assert manifest["nodes"] == 3 and manifest["node_id_policy"] == "sorted-compact"
	assert np.fromfile(destination / "original_ids.i64", dtype="<i8").tolist() == [7, 100, 900]
	assert np.fromfile(destination / "sources.i64", dtype="<i8").tolist() == [1, 1, 2]
	assert np.fromfile(destination / "targets.i64", dtype="<i8").tolist() == [0, 0, 2]
	with pytest.raises(ValueError, match="distinct nodes"):
		prepare_edges(source, tmp_path / "bad-compact", nodes=4, compact_ids=True)


def test_snap_reddit_conversion_uses_local_full_graph_without_dgl_download(tmp_path, monkeypatch):
	import io
	import zipfile

	from scripts.download_datasets import _reddit

	features = io.BytesIO()
	np.save(features, np.arange(6, dtype=np.float32).reshape(3, 2))
	with zipfile.ZipFile(tmp_path / "reddit.zip", "w") as stream:
		for name, value in {
			"reddit-id_map.json": {"10": 0, "20": 1, "30": 2},
			"reddit-class_map.json": {"10": 0, "20": 1, "30": 0},
			"reddit-G_full.json": {
				"directed": False,
				"nodes": [{"id": 10}, {"id": 20, "val": True}, {"id": 30, "test": True}],
				"links": [{"source": 10, "target": 20}, {"source": 30, "target": 30}],
			},
		}.items():
			stream.writestr("reddit/" + name, json.dumps(value))
		stream.writestr("reddit/reddit-feats.npy", features.getvalue())
	monkeypatch.setattr("scripts.download_datasets.download", lambda *a, **k: {"sha256": "fixture"})
	record = _reddit(tmp_path, url="unused", connections=1)
	assert record.nodes == 3 and record.edges == 3 and record.features == 2
	graph = load_graph(tmp_path / "reddit" / "dgl_graph.bin")
	assert graph.ndata["label"].tolist() == [0, 1, 0]
	assert graph.ndata["test_mask"].tolist() == [False, False, True]
	assert _reddit(tmp_path, url="unused", connections=1).edges == 3
	conversion = json.loads((tmp_path / "reddit" / "conversion-manifest.json").read_text())
	assert conversion["archive_sha256"] == "fixture" and conversion["graph_sha256"]
	monkeypatch.setattr(
		"scripts.download_datasets.download",
		lambda *a, **k: {"sha256": "changed", "bytes": conversion["archive_bytes"]},
	)
	with pytest.raises(ValueError, match="archive_sha256"):
		_reddit(tmp_path, url="unused", connections=1)


def test_reddit_does_not_relabel_an_existing_dgl_graph_without_conversion_metadata(tmp_path, monkeypatch):
	import dgl

	from scripts.download_datasets import _reddit

	folder = tmp_path / "reddit"
	folder.mkdir()
	graph = dgl.graph(([0], [1]), num_nodes=2)
	dgl.save_graphs(str(folder / "dgl_graph.bin"), [graph])
	monkeypatch.setattr("scripts.download_datasets.download", lambda *a, **k: {"sha256": "archive", "bytes": 100})
	with pytest.raises(ValueError, match="conversion metadata is missing"):
		_reddit(tmp_path, url="unused", connections=1)


def test_reddit_converted_graph_requires_matching_dimensions_labels_and_splits():
	import dgl

	from scripts.download_datasets import _validate_reddit_graph

	metadata = {"nodes": 3, "edges": 2, "feature_width": 2}
	graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
	graph.ndata["feat"] = torch.ones(3, 2, dtype=torch.float32)
	graph.ndata["label"] = torch.tensor([0, 1, 0], dtype=torch.int64)
	graph.ndata["train_mask"] = torch.tensor([True, False, False])
	graph.ndata["val_mask"] = torch.tensor([False, True, False])
	graph.ndata["test_mask"] = torch.tensor([False, False, True])
	assert _validate_reddit_graph(graph, metadata=metadata) == {"nodes": 3, "edges": 2, "feature_width": 2}

	bad_dimensions = dict(metadata, edges=3)
	with pytest.raises(ValueError, match="dimensions"):
		_validate_reddit_graph(graph, metadata=bad_dimensions)
	graph.ndata["label"] = torch.tensor([0, -1, 0], dtype=torch.int64)
	with pytest.raises(ValueError, match="nonnegative"):
		_validate_reddit_graph(graph, metadata=metadata)
	graph.ndata["label"] = torch.tensor([0, 1, 0], dtype=torch.int64)
	graph.ndata["val_mask"][0] = True
	with pytest.raises(ValueError, match="partition"):
		_validate_reddit_graph(graph, metadata=metadata)


def _friendster_fixture(root, *, feature_dim=2, seed=7, source_sha256=None, undirected=True):
	from lib.dataset.download import file_sha256

	root.mkdir(parents=True, exist_ok=True)
	raw = root / "raw" / "edges.txt.gz"
	raw.parent.mkdir(parents=True, exist_ok=True)
	with gzip.open(raw, "wt") as stream:
		stream.write("0 1\n1 2\n2 2\n")
	sha256 = source_sha256 or file_sha256(raw)
	prepared = root / "prepared"
	manifest = prepare_edges(
		raw,
		prepared,
		nodes=3,
		feature_dim=feature_dim,
		seed=seed,
		chunk_edges=2,
		expected_edges=3,
		source_sha256=sha256,
		compact_ids=True,
		undirected=undirected,
	)
	return raw, prepared, sha256, manifest


def _reuse_friendster(root, monkeypatch, *, feature_dim=2, seed=7, expected_edges=3):
	from scripts.download_datasets import _friendster

	raw = root / "raw" / "edges.txt.gz"
	sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
	monkeypatch.setattr(
		"scripts.download_datasets.download",
		lambda *args, **kwargs: {"sha256": sha256, "bytes": raw.stat().st_size},
	)
	return _friendster(
		root,
		url="fixture",
		nodes=3,
		expected_edges=expected_edges,
		feature_dim=feature_dim,
		connections=1,
		chunk_edges=2,
		seed=seed,
	)


def test_friendster_reuses_only_matching_complete_undirected_preparation(tmp_path, monkeypatch):
	_, prepared, _, manifest = _friendster_fixture(tmp_path)
	assert manifest["input_edges"] == 3 and manifest["edges"] == 5
	before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in prepared.iterdir() if path.is_file()}
	monkeypatch.setattr(
		"scripts.download_datasets.prepare_edges",
		lambda *args, **kwargs: pytest.fail("valid prepared data must be reused"),
	)
	record = _reuse_friendster(tmp_path, monkeypatch)
	after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in prepared.iterdir() if path.is_file()}
	assert record.status == "ready" and record.edges == 5
	assert before == after


def test_friendster_rejects_a_prepared_manifest_for_a_different_source_hash(tmp_path, monkeypatch):
	_friendster_fixture(tmp_path, source_sha256="0" * 64)
	with pytest.raises(ValueError, match="source_sha256"):
		_reuse_friendster(tmp_path, monkeypatch)


def test_friendster_rejects_old_directed_preparation(tmp_path, monkeypatch):
	_friendster_fixture(tmp_path, undirected=False)
	with pytest.raises(ValueError, match="source_undirected"):
		_reuse_friendster(tmp_path, monkeypatch)


@pytest.mark.parametrize(
	("stored_width", "stored_seed", "requested_width", "requested_seed"), [(2, 7, 3, 7), (2, 7, 2, 8)]
)
def test_friendster_rejects_feature_configuration_mismatch(
	tmp_path, monkeypatch, stored_width, stored_seed, requested_width, requested_seed
):
	_friendster_fixture(tmp_path, feature_dim=stored_width, seed=stored_seed)
	with pytest.raises(ValueError, match=r"features\."):
		_reuse_friendster(tmp_path, monkeypatch, feature_dim=requested_width, seed=requested_seed)


@pytest.mark.parametrize("damage", ["truncate-feature-header", "truncate-edge-file", "remove-original-ids"])
def test_friendster_rejects_incomplete_prepared_outputs(tmp_path, monkeypatch, damage):
	_, prepared, _, _ = _friendster_fixture(tmp_path)
	if damage == "truncate-feature-header":
		path = prepared / "features.npy"
		path.write_bytes(path.read_bytes()[:-1])
	elif damage == "truncate-edge-file":
		path = prepared / "targets.i64"
		path.write_bytes(path.read_bytes()[:-1])
	else:
		(prepared / "original_ids.i64").unlink()
	with pytest.raises(ValueError, match="incomplete|size|truncated"):
		_reuse_friendster(tmp_path, monkeypatch)


def test_undirected_preparation_expands_arcs_and_checks_input_edge_count(tmp_path):
	source = tmp_path / "undirected.txt"
	source.write_text("100 7\n100 7\n900 900\n")
	destination = tmp_path / "symmetric"
	manifest = prepare_edges(
		source, destination, nodes=3, feature_dim=2, chunk_edges=1, compact_ids=True, undirected=True, expected_edges=3
	)
	assert manifest["input_edges"] == 3 and manifest["edges"] == 5
	assert manifest["self_loops"] == 1 and manifest["source_undirected"]
	assert np.fromfile(destination / "sources.i64", dtype="<i8").tolist() == [1, 0, 1, 0, 2]
	assert np.fromfile(destination / "targets.i64", dtype="<i8").tolist() == [0, 1, 0, 1, 2]


def test_snap_reddit_positional_endpoints_follow_node_order_then_feature_id_map(tmp_path, monkeypatch):
	import io
	import zipfile

	from scripts.download_datasets import _reddit

	features = io.BytesIO()
	np.save(features, np.arange(6, dtype=np.float32).reshape(3, 2))
	with zipfile.ZipFile(tmp_path / "reddit.zip", "w") as stream:
		for name, value in {
			"reddit-id_map.json": {"alice": 2, "bob": 0, "carol": 1},
			"reddit-class_map.json": {"alice": 0, "bob": 1, "carol": 0},
			"reddit-G_full.json": {
				"directed": False,
				"nodes": [{"id": "alice"}, {"id": "bob", "val": True}, {"id": "carol", "test": True}],
				"links": [{"source": 0, "target": 1}, {"source": 2, "target": 2}],
			},
		}.items():
			stream.writestr("reddit/" + name, json.dumps(value))
		stream.writestr("reddit/reddit-feats.npy", features.getvalue())
	monkeypatch.setattr("scripts.download_datasets.download", lambda *a, **k: {"sha256": "fixture"})
	_reddit(tmp_path, url="unused", connections=1)
	graph = load_graph(tmp_path / "reddit/dgl_graph.bin")
	u, v = graph.edges()
	assert u.tolist() == [2, 1, 0] and v.tolist() == [0, 1, 2]
	assert graph.ndata["label"].tolist() == [1, 0, 0]
	assert graph.ndata["test_mask"].tolist() == [False, True, False]
	metadata = json.loads((tmp_path / "reddit/conversion-manifest.json").read_text())
	assert metadata["endpoint_encoding"] == "positions"
