import shutil
from multiprocessing.connection import Listener
from threading import Thread

import dgl
import numpy as np
import pytest

from lib.storage.base import BaseTopology
from lib.storage.capacity import GIB, StorageNodeBudget
from lib.storage.remote import RemoteStorage, base_fingerprint
from scripts.storage_server import serve, validate_init

pytestmark = pytest.mark.host


def test_remote_storage_routes_rows_and_mutations_over_authenticated_connection(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	base = BaseTopology.export(graph, tmp_path / "base")
	authkey = b"test-key"
	listener = Listener(("127.0.0.1", 0), authkey=authkey)
	endpoint = f"127.0.0.1:{listener.address[1]}"

	def serve_one(connection):
		try:
			assert connection.recv() == ("ping",)
			connection.send((True, "ready"))
			command, *_ = connection.recv()
			assert command == "init"
			serve(
				connection,
				base=base,
				start=0,
				stop=4,
				overlay_bytes=1024,
				response_bytes=1024,
				profiling=True,
			)
		finally:
			connection.close()

	def worker():
		threads = []
		for _ in range(2):
			thread = Thread(target=serve_one, args=(listener.accept(),))
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join(timeout=5)
		listener.close()

	thread = Thread(target=worker)
	thread.start()
	storage = RemoteStorage(
		base,
		(endpoint,),
		budgets=(StorageNodeBudget("node", capacity_bytes=GIB, reserve_bytes=0),),
		authkey=authkey,
		overlay_bytes=1024,
		response_bytes=1024,
		timeout=5,
	)
	try:
		nodes = np.array([0, 2], dtype=np.int64)
		result = storage.exchange({0: ("rows", "out", nodes)})[0]
		assert result[0] == "remote-rows"
		np.testing.assert_array_equal(result[1], np.array([1, 1], dtype=np.int64))
		np.testing.assert_array_equal(result[2], np.array([1, 3], dtype=np.int64))
		storage.exchange(
			{
				0: (
					"prepare",
					(np.array([0], dtype=np.int64), np.array([3], dtype=np.int64)),
					(np.array([3], dtype=np.int64), np.array([0], dtype=np.int64)),
					False,
					1,
				)
			}
		)
		storage.exchange({0: ("commit", 1)})
		assert storage.stats()[0]["revision"] == 1
	finally:
		storage.close()
	thread.join(timeout=5)
	assert not thread.is_alive()


def test_prestarted_server_validates_path_independent_base_and_serves_rows(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	client_folder = tmp_path / "client-base"
	client_base = BaseTopology.export(graph, client_folder)
	server_folder = tmp_path / "server-base"
	shutil.copytree(client_folder, server_folder)
	server_base = BaseTopology(server_folder, graph.num_nodes(), graph.num_edges())
	authkey = b"integration-key"
	listener = Listener(("127.0.0.1", 0), authkey=authkey)
	endpoint = f"127.0.0.1:{listener.address[1]}"

	def serve_one(connection):
		try:
			assert connection.recv() == ("ping",)
			connection.send((True, "ready"))
			command, *args = connection.recv()
			assert command == "init"
			validate_init(
				args,
				base=server_base,
				start=0,
				stop=4,
				capacity_bytes=GIB,
				reserve_bytes=0,
				overlay_bytes=1024,
				response_bytes=1024,
			)
			serve(
				connection,
				base=server_base,
				start=0,
				stop=4,
				overlay_bytes=1024,
				response_bytes=1024,
				profiling=False,
			)
		finally:
			connection.close()

	def worker():
		threads = []
		for _ in range(2):
			thread = Thread(target=serve_one, args=(listener.accept(),))
			thread.start()
			threads.append(thread)
		for thread in threads:
			thread.join(timeout=5)
		listener.close()

	thread = Thread(target=worker)
	thread.start()
	storage = RemoteStorage(
		client_base,
		(endpoint,),
		budgets=(StorageNodeBudget("node", capacity_bytes=GIB, reserve_bytes=0),),
		authkey=authkey,
		overlay_bytes=1024,
		response_bytes=1024,
		timeout=5,
	)
	try:
		result = storage.exchange({0: ("rows", "out", np.array([0, 2], dtype=np.int64))})[0]
		np.testing.assert_array_equal(result[2], np.array([1, 3], dtype=np.int64))
	finally:
		storage.close()
	thread.join(timeout=5)
	assert not thread.is_alive()
	assert client_folder.exists() and server_folder.exists()
	assert base_fingerprint(client_base) == base_fingerprint(server_base)


def test_validate_init_rejects_wrong_same_dimension_graph_and_budget(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	wrong = dgl.graph(([0, 1, 3], [1, 2, 2]), num_nodes=4)
	base = BaseTopology.export(graph, tmp_path / "base")
	wrong_base = BaseTopology.export(wrong, tmp_path / "wrong")

	def args_for(fingerprint, *, capacity=GIB, overlay=1024):
		return [
			"/host-specific/path",
			4,
			3,
			0,
			4,
			overlay,
			1024,
			capacity,
			0,
			fingerprint,
		]

	with pytest.raises(ValueError, match="graph content"):
		validate_init(
			args_for(base_fingerprint(wrong_base)),
			base=base,
			start=0,
			stop=4,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)
	with pytest.raises(ValueError, match="node budget"):
		validate_init(
			args_for(base_fingerprint(base), capacity=GIB + 1),
			base=base,
			start=0,
			stop=4,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)
	wrong_partition = args_for(base_fingerprint(base))
	wrong_partition[4] = 3
	with pytest.raises(ValueError, match="topology partition"):
		validate_init(
			wrong_partition,
			base=base,
			start=0,
			stop=4,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)
	wrong_dimensions = args_for(base_fingerprint(base))
	wrong_dimensions[1] = 5
	with pytest.raises(ValueError, match="topology partition"):
		validate_init(
			wrong_dimensions,
			base=base,
			start=0,
			stop=4,
			capacity_bytes=GIB,
			reserve_bytes=0,
			overlay_bytes=1024,
			response_bytes=1024,
		)


def test_server_cli_accepts_sampling_while_control_remains_open(tmp_path):
	import json
	import subprocess
	import sys

	graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
	base = BaseTopology.export(graph, tmp_path / "base")
	process = subprocess.Popen(
		[
			sys.executable,
			"-m",
			"scripts.storage_server",
			"--base",
			str(base.folder),
			"--nodes",
			"3",
			"--edges",
			"2",
			"--start",
			"0",
			"--stop",
			"3",
			"--host",
			"127.0.0.1",
			"--port",
			"0",
			"--capacity-bytes",
			str(GIB),
			"--reserve-bytes",
			"0",
			"--overlay-bytes",
			"1024",
			"--response-bytes",
			"1024",
		],
		stdout=subprocess.PIPE,
		text=True,
	)
	try:
		address = json.loads(process.stdout.readline())["listen"]
		storage = RemoteStorage(
			base,
			(f"127.0.0.1:{address[1]}",),
			budgets=(StorageNodeBudget("node", capacity_bytes=GIB, reserve_bytes=0),),
			authkey=b"neutronrt",
			overlay_bytes=1024,
			response_bytes=1024,
			timeout=5,
		)
		try:
			assert storage.sample_rss()[0]["rss_bytes"] > 0
			assert storage.stats()[0]["revision"] == 0
		finally:
			storage.close()
	finally:
		process.terminate()
		process.wait(timeout=5)
