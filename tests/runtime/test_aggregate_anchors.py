from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.ownership import MissingState, StateKey
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.transport import ReadyResult
from tests.runtime.test_prepared import _actors
from tests.runtime.test_reconstruction_worker import _state


@torch.inference_mode()
def test_old_sum_anchor_survives_nested_retirement_and_storage_eviction_without_value_copy():
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.ones(6, 4)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		_, sums = _state(plane, workers, graph, features)
		key, location, expected = sums[0]
		worker = workers[0]
		worker.store.max_bytes = 96
		transport.endpoints[worker.owner].max_bytes = 96
		transport.endpoints[worker.owner].bytes_used = 96
		with plane.retain_aggregates([key, key]):
			with worker.store.acquire(location.handle, wait=True) as value:
				assert value.data_ptr() == expected.data_ptr()
			with plane.retain_aggregates([key]):
				plane.retire(key)
				plane.retire(key)
			assert plane.directory.locate(key) == location
			# A retained sum may still move to storage under pressure. The version
			# anchor follows the key rather than pinning a GPU allocation forever.
			plane.make_room(worker.owner, 96)
			assert plane.directory.locate(key).owner == plane.storage.owner
			with ExitStack() as leases:
				torch.testing.assert_close(plane.read_rows(key, torch.arange(6), leases), expected)
		with pytest.raises(MissingState):
			plane.directory.locate(key)
		assert plane.storage.store.bytes_used == 0
		assert not plane._retained_states and not plane._pending_retire


@torch.inference_mode()
def test_normal_delta_mode_can_retain_old_h_while_aggregate_only_rejects_h_anchors():
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		worker = workers[0]
		values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
		handle = worker.store.put(values)
		worker.store.mark_published(handle)
		key = StateKey(0, 1, 100)
		attempt = plane.directory.begin(key, worker.owner)
		plane.publish_result(attempt, ReadyResult(handle, 0, 1, 100, (6, 4), 96))
		transport.endpoints[worker.owner].max_bytes = 96
		transport.endpoints[worker.owner].bytes_used = 96
		with plane.retain_states([key]):
			plane.retire(key)
			plane.make_room(worker.owner, 96)
			assert plane.directory.locate(key).owner == plane.storage.owner
			with ExitStack() as leases:
				torch.testing.assert_close(plane.read_rows(key, torch.arange(6), leases), values)
		with pytest.raises(MissingState):
			plane.directory.locate(key)
		plane.discard_embeddings = True
		with pytest.raises(ValueError, match="historical h"), plane.retain_states([key]):
			pass
		assert not plane._retained_states and not plane._pending_retire


@torch.inference_mode()
def test_anchor_validation_is_atomic_and_exception_releases_deferred_retirement():
	graph = dgl.graph(([], []), num_nodes=6)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		_, sums = _state(plane, workers, graph, torch.ones(6, 4))
		key = sums[0][0]
		with pytest.raises(MissingState), plane.retain_aggregates([key, StateKey(9, 1, 999, "a")]):
			pass
		assert not plane._retained_states
		with pytest.raises(ValueError, match="sum state"), plane.retain_aggregates([StateKey(0, 1, 1)]):
			pass
		with pytest.raises(RuntimeError, match="consumer"), plane.retain_aggregates([key]):
			plane.retire(key)
			raise RuntimeError("consumer failed")
		with pytest.raises(MissingState):
			plane.directory.locate(key)
		assert not plane._retained_states and not plane._pending_retire
