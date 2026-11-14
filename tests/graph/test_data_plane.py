from threading import Event

import pytest
import torch

from lib.graph.data_plane import LocalEmbeddingPlane, RejectedPublication, copy_tensor
from lib.graph.ownership import LastReplica, Location, MissingState, StateKey
from lib.graph.resident import BufferInUse


def test_compute_publication_keeps_results_on_the_producer_without_storage_copy():
	plane = LocalEmbeddingPlane()
	try:
		storage = plane.add_node(0, "cpu", 64)
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		value = torch.arange(4.0)
		attempt = plane.directory.begin(key, worker.owner)
		location = plane.publish(attempt, value)
		assert storage.store.bytes_used == 0
		with plane.acquire(key) as result:
			assert result.data_ptr() == value.data_ptr()
		assert plane.directory.locate(key) == location
		old = plane.directory.begin(StateKey(1, 1, 0), worker.owner)
		plane.directory.begin(old.key, worker.owner)
		with pytest.raises(RejectedPublication):
			plane.publish(old, torch.ones(4))
		assert worker.store.bytes_used == 16
	finally:
		plane.close()


def test_writeback_is_deferred_deduplicated_and_bounded():
	started, proceed = Event(), Event()

	def transfer(value, device):
		started.set()
		assert proceed.wait(10)
		return copy_tensor(value, device)

	plane = LocalEmbeddingPlane(max_pending=1, transfer=transfer)
	try:
		storage = plane.add_node(0, "cpu", 64)
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		plane.publish(plane.directory.begin(key, worker.owner), torch.arange(4.0))
		future = plane.writeback(key, storage.owner)
		assert started.wait(5)
		assert not future.done()
		assert storage.store.bytes_used == 0
		assert plane.writeback(key, storage.owner) is future
		with plane.acquire(key) as value:
			torch.testing.assert_close(value, torch.arange(4.0))
		other = StateKey(0, 1, 1)
		plane.publish(plane.directory.begin(other, worker.owner), torch.ones(4))
		with pytest.raises(MemoryError, match="queue is full"):
			plane.writeback(other, storage.owner)
		proceed.set()
		location = future.result(timeout=5)
		assert location.owner == storage.owner
		assert plane.writeback(key, storage.owner).result(timeout=5) == location
		with plane.acquire(key, owner=storage.owner) as value:
			torch.testing.assert_close(value, torch.arange(4.0))
	finally:
		proceed.set()
		plane.close()


def test_owner_loss_during_copy_rejects_late_writeback_and_releases_partial_state():
	started, proceed = Event(), Event()

	def transfer(value, device):
		started.set()
		assert proceed.wait(10)
		return copy_tensor(value, device)

	plane = LocalEmbeddingPlane(transfer=transfer)
	try:
		storage = plane.add_node(0, "cpu", 64)
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		plane.publish(plane.directory.begin(key, worker.owner), torch.arange(4.0))
		future = plane.writeback(key, storage.owner)
		assert started.wait(5)
		assert plane.lose_owner(worker.owner) == (key,)
		restarted = plane.add_node(1, "cpu", 64)
		plane.publish(plane.directory.begin(key, restarted.owner), torch.arange(4.0) + 10)
		proceed.set()
		with pytest.raises(RejectedPublication):
			future.result(timeout=5)
		assert storage.store.bytes_used == 0
		assert plane.directory.locate(key).owner == restarted.owner
		with plane.acquire(key) as value:
			torch.testing.assert_close(value, torch.arange(4.0) + 10)
	finally:
		proceed.set()
		plane.close()


def test_active_readers_and_loss_of_backup_prevent_unsafe_eviction():
	plane = LocalEmbeddingPlane()
	try:
		storage = plane.add_node(0, "cpu", 64)
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		location = plane.publish(plane.directory.begin(key, worker.owner), torch.arange(4.0))
		plane.writeback(key, storage.owner).result(timeout=5)
		with plane.acquire(key, owner=worker.owner), pytest.raises(BufferInUse):
			plane.evict(key, worker.owner)
		assert plane.directory.locate(key, preferred=worker.owner) == location
		plane.lose_owner(storage.owner)
		with pytest.raises(LastReplica):
			plane.evict(key, worker.owner)
		assert worker.store.bytes_used == 16
		restarted = plane.add_node(0, "cpu", 64)
		plane.writeback(key, restarted.owner).result(timeout=5)
		plane.evict(key, worker.owner)
		assert worker.store.bytes_used == 0
		assert plane.directory.locate(key).owner == restarted.owner
	finally:
		plane.close()


def test_transfer_failure_preserves_the_published_source():
	def fail(*_args):
		raise RuntimeError("injected transfer failure")

	plane = LocalEmbeddingPlane(transfer=fail)
	try:
		storage = plane.add_node(0, "cpu", 64)
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		location = plane.publish(plane.directory.begin(key, worker.owner), torch.arange(4.0))
		with pytest.raises(RuntimeError, match="injected"):
			plane.writeback(key, storage.owner).result(timeout=5)
		assert plane.directory.locate(key) == location
		assert storage.store.bytes_used == 0
	finally:
		plane.close()
	assert plane.failed_transfers == 1


def test_retirement_reclaims_unflushed_state_without_invalidating_active_readers():
	plane = LocalEmbeddingPlane()
	try:
		worker = plane.add_node(1, "cpu", 64)
		key = StateKey(0, 1, 0)
		attempt = plane.directory.begin(key, worker.owner)
		plane.publish(attempt, torch.arange(4.0))
		lease = plane.acquire(key)
		plane.retire(key)
		with pytest.raises(MissingState):
			plane.acquire(key)
		torch.testing.assert_close(lease.tensor, torch.arange(4.0))
		assert worker.store.bytes_used == 16
		lease.close()
		assert worker.store.bytes_used == 0
		assert not plane.directory.publish(attempt, 9)
		plane.retire(key)
	finally:
		plane.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs")
def test_gpu_peer_copy_and_deferred_storage_writeback():
	plane = LocalEmbeddingPlane()
	try:
		storage = plane.add_node(0, "cpu", 4096)
		first = plane.add_node(1, "cuda:0", 4096)
		second = plane.add_node(2, "cuda:1", 4096)
		key = StateKey(0, 1, 0)
		value = torch.arange(1024, dtype=torch.float32, device="cuda:0")
		plane.publish(plane.directory.begin(key, first.owner), value)
		assert storage.store.bytes_used == 0
		with plane.acquire(key, owner=second.owner) as result:
			assert result.device == torch.device("cuda:1")
			torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32))
		cpu = plane.writeback(key, storage.owner).result(timeout=10)
		assert cpu.owner == storage.owner
		plane.lose_owner(first.owner)
		plane.lose_owner(second.owner)
		restarted = plane.add_node(1, "cuda:0", 4096)
		with plane.acquire(key, owner=restarted.owner) as result:
			assert result.device == torch.device("cuda:0")
			torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32))
		assert plane.directory.locate(key, preferred=storage.owner) == Location(storage.owner, cpu.handle)
		lease = plane.acquire(key, owner=restarted.owner)
		plane.retire(key)
		torch.testing.assert_close(lease.tensor.cpu(), torch.arange(1024, dtype=torch.float32))
		assert storage.store.bytes_used == 0
		assert restarted.store.bytes_used == 4096
		lease.close()
		assert restarted.store.bytes_used == 0
	finally:
		plane.close()


def test_required_fetch_bypasses_unrelated_blocked_writeback():
	started, proceed = Event(), Event()

	def transfer(value, device):
		if float(value[0]) == 0:
			started.set()
			assert proceed.wait(10)
		return copy_tensor(value, device)

	plane = LocalEmbeddingPlane(max_pending=1, transfer=transfer)
	try:
		storage = plane.add_node(0, "cpu", 128)
		producer = plane.add_node(1, "cpu", 128)
		consumer = plane.add_node(2, "cpu", 128)
		background, required = StateKey(0, 1, 0), StateKey(0, 1, 1)
		plane.publish(plane.directory.begin(background, producer.owner), torch.arange(4.0))
		plane.publish(plane.directory.begin(required, producer.owner), torch.arange(4.0) + 17)
		future = plane.writeback(background, storage.owner)
		assert started.wait(5)
		with plane.acquire(required, owner=consumer.owner) as values:
			torch.testing.assert_close(values, torch.arange(4.0) + 17)
		assert not future.done()
		assert not proceed.is_set()
		proceed.set()
		future.result(timeout=5)
	finally:
		proceed.set()
		plane.close()


def test_foreground_fetch_failure_keeps_the_source_and_can_be_retried():
	calls = 0

	def transfer(value, device):
		nonlocal calls
		calls += 1
		if calls == 1:
			raise RuntimeError("required fetch failed")
		return copy_tensor(value, device)

	plane = LocalEmbeddingPlane(transfer=transfer)
	try:
		producer = plane.add_node(1, "cpu", 64)
		consumer = plane.add_node(2, "cpu", 64)
		key = StateKey(0, 1, 0)
		plane.publish(plane.directory.begin(key, producer.owner), torch.arange(4.0))
		with pytest.raises(RuntimeError, match="required fetch failed"):
			plane.acquire(key, owner=consumer.owner)
		assert consumer.store.bytes_used == 0
		assert plane.directory.locate(key).owner == producer.owner
		with plane.acquire(key, owner=consumer.owner) as values:
			torch.testing.assert_close(values, torch.arange(4.0))
		assert plane.failed_transfers == 1
	finally:
		plane.close()
