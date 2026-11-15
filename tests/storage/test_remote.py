from multiprocessing.connection import Listener
from threading import Thread

import dgl
import numpy as np
import pytest

from lib.storage.base import BaseTopology
from lib.storage.capacity import GIB, StorageNodeBudget
from lib.storage.remote import RemoteStorage
from scripts.storage_server import serve

pytestmark = pytest.mark.host


def test_remote_storage_routes_rows_and_mutations_over_authenticated_connection(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	base = BaseTopology.export(graph, tmp_path / "base")
	authkey = b"test-key"
	listener = Listener(("127.0.0.1", 0), authkey=authkey)
	endpoint = f"127.0.0.1:{listener.address[1]}"

	def worker():
		connection = listener.accept()
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
