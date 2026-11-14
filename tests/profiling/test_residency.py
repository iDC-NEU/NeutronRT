from contextlib import ExitStack

import dgl
import torch

from lib.graph.ownership import StateKey
from lib.profiling.residency import retained_state
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.transport import ReadyResult
from tests.runtime.test_prepared import _actors
from tests.runtime.test_reconstruction_worker import _state


@torch.inference_mode()
def test_resident_snapshot_counts_state_kind_and_each_live_replica_without_copying_values():
	graph = dgl.graph(([], []), num_nodes=6)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		outputs, sums = _state(plane, workers, graph, torch.ones(6, 4))
		worker = workers[0]
		handle = worker.store.put(outputs[0])
		worker.store.mark_published(handle)
		h = plane.directory.begin(StateKey(0, 1, 99), worker.owner)
		plane.publish_result(h, ReadyResult(handle, 0, 1, 99, (6, 4), worker.store.bytes_used))
		first = retained_state(plane)
		by_rank = {row["rank"]: row for row in first["owners"]}
		assert by_rank[0]["h_bytes"] == by_rank[0]["aggregate_bytes"] == first["storage_resident_bytes"] == 0
		assert by_rank[1]["h_bytes"] == by_rank[1]["aggregate_bytes"] == 96
		assert by_rank[2]["aggregate_bytes"] == 72 and by_rank[2]["h_bytes"] == 0
		plane.writeback(sums[0][0], plane.storage.owner).result(5)
		second = retained_state(plane)
		by_rank = {row["rank"]: row for row in second["owners"]}
		assert by_rank[0]["aggregate_bytes"] == by_rank[1]["aggregate_bytes"] == 96
		assert second["storage_resident_bytes"] == 96
		plane.evict(sums[0][0], worker.owner)
		plane.retire(h.key)
		third = retained_state(plane)
		by_rank = {row["rank"]: row for row in third["owners"]}
		assert by_rank[1]["h_bytes"] == by_rank[1]["aggregate_bytes"] == 0
		assert by_rank[0]["aggregate_bytes"] == third["storage_resident_bytes"] == 96
		# Earlier reports own plain integer counters, not live references to metadata.
		assert first["owners"][1]["h_bytes"] == 96
