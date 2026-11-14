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
