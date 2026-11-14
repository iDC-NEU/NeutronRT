import gc

import pytest
import torch

from lib.graph.resident import BufferInUse, BufferNotReady, ResidentTensors, UnflushedBuffer


class Readiness:
	def __init__(self):
		self.done = False
		self.waited = 0

	def query(self):
		return self.done

	def synchronize(self):
		self.waited += 1
		self.done = True


def test_resident_buffer_retains_storage_without_copy_or_autograd_history():
	store = ResidentTensors(64)
	value = torch.arange(8.0, requires_grad=True)
	pointer = value.data_ptr()
	handle = store.put(value)
	del value
	gc.collect()
	with store.acquire(handle) as tensor:
		assert tensor.data_ptr() == pointer
		assert not tensor.requires_grad
		torch.testing.assert_close(tensor, torch.arange(8.0))
	assert store.bytes_used == 32


def test_views_charge_their_shared_backing_storage_once():
	store = ResidentTensors(64)
	base = torch.arange(16.0)
	first = store.put(base[::2])
	second = store.put(base[1::2])
	assert store.bytes_used == 64
	with pytest.raises(MemoryError, match="budget exhausted"):
		store.put(torch.ones(1))
	store.mark_backed_up(first)
	store.evict(first)
	assert store.bytes_used == 64
	with store.acquire(second) as tensor:
		assert tensor.data_ptr() == base[1:].data_ptr()
		torch.testing.assert_close(tensor, base[1::2])
	store.mark_backed_up(second)
	store.evict(second)
	assert store.bytes_used == 0


def test_budget_counts_full_storage_even_for_tiny_views():
	store = ResidentTensors(4)
	with pytest.raises(MemoryError):
		store.put(torch.zeros(16)[:1])
	assert store.bytes_used == 0


def test_unready_buffer_cannot_be_read_and_failed_acquire_does_not_pin_it():
	store = ResidentTensors(16)
	ready = Readiness()
	handle = store.put(torch.ones(4), readiness=ready)
	assert not store.is_ready(handle)
	with pytest.raises(BufferNotReady):
		store.acquire(handle)
	with store.acquire(handle, wait=True) as tensor:
		assert ready.waited == 1
		assert store.is_ready(handle)
		torch.testing.assert_close(tensor, torch.ones(4))
	store.mark_backed_up(handle)
	store.evict(handle)
	assert store.bytes_used == 0


def test_eviction_requires_backup_and_completion_of_all_consumers():
	store = ResidentTensors(16)
	handle = store.put(torch.ones(4))
	with pytest.raises(UnflushedBuffer):
		store.evict(handle)
	first = store.acquire(handle)
	second = store.acquire(handle)
	store.mark_backed_up(handle)
	with pytest.raises(BufferInUse):
		store.evict(handle)
	first.close()
	first.close()
	with pytest.raises(BufferInUse):
		store.evict(handle)
	second.close()
	store.evict(handle)
	with pytest.raises(KeyError):
		store.acquire(handle)
	with pytest.raises(RuntimeError, match="closed"), first:
		pass
	assert store.put(torch.ones(4)) > handle


def test_consumer_exception_releases_lease():
	store = ResidentTensors(16)
	handle = store.put(torch.ones(4))
	store.mark_backed_up(handle)
	with pytest.raises(ValueError, match="consumer"), store.acquire(handle):
		raise ValueError("consumer")
	store.evict(handle)
	assert store.bytes_used == 0


def test_zero_budget_allows_empty_results_only():
	store = ResidentTensors(0)
	handle = store.put(torch.empty(0))
	with store.acquire(handle) as tensor:
		assert tensor.numel() == 0
	with pytest.raises(MemoryError):
		store.put(torch.ones(1))
	with pytest.raises(ValueError, match="nonnegative"):
		ResidentTensors(-1)
	with pytest.raises(ValueError, match="strided"):
		store.put(torch.sparse_coo_tensor([[0]], [1.0], (2,)))


def test_cancelled_outputs_can_be_freed_but_published_results_remain_owned():
	store = ResidentTensors(16)
	handle = store.put(torch.ones(4))
	lease = store.acquire(handle)
	with pytest.raises(BufferInUse):
		store.abandon(handle)
	lease.close()
	store.abandon(handle)
	assert store.bytes_used == 0
	handle = store.put(torch.ones(4))
	store.mark_published(handle)
	store.mark_published(handle)
	with pytest.raises(UnflushedBuffer, match="Published"):
		store.abandon(handle)
	store.mark_backed_up(handle)
	store.evict(handle)
	assert store.bytes_used == 0


def test_publication_requires_compute_readiness():
	store = ResidentTensors(16)
	ready = Readiness()
	handle = store.put(torch.ones(4), readiness=ready)
	with pytest.raises(BufferNotReady):
		store.mark_published(handle)
	with store.acquire(handle, wait=True):
		store.mark_published(handle)
	with pytest.raises(UnflushedBuffer):
		store.abandon(handle)


def test_related_outputs_publish_atomically_after_every_readiness_fence():
	store = ResidentTensors(32)
	ready = Readiness()
	first, second = store.put(torch.ones(4)), store.put(torch.ones(4), readiness=ready)
	with pytest.raises(BufferNotReady):
		store.mark_published_many((first, second))
	store.abandon(first)  # The ready first member was not partially published.
	first = store.put(torch.ones(4))
	with store.acquire(second, wait=True):
		store.mark_published_many((first, second))
	for handle in (first, second):
		with pytest.raises(UnflushedBuffer):
			store.abandon(handle)
		store.retire(handle)
	assert store.bytes_used == 0


def test_invalid_or_retired_group_member_cannot_publish_other_outputs():
	store = ResidentTensors(32)
	first, second = store.put(torch.ones(4)), store.put(torch.ones(4))
	with pytest.raises(KeyError):
		store.mark_published_many((first, 999))
	lease = store.acquire(second)
	store.retire(second)
	with pytest.raises(BufferInUse, match="retired"):
		store.mark_published_many((first, second))
	store.abandon(first)
	lease.close()
	assert store.bytes_used == 0


def test_eviction_authorization_runs_before_release_and_can_reject():
	store = ResidentTensors(16)
	handle = store.put(torch.ones(4))
	store.mark_backed_up(handle)
	called = []

	def reject():
		called.append(True)
		raise ValueError("Replica is still needed")

	with store.acquire(handle), pytest.raises(BufferInUse):
		store.evict(handle, authorize=reject)
	assert not called
	with pytest.raises(ValueError, match="still needed"):
		store.evict(handle, authorize=reject)
	assert store.bytes_used == 16
	store.evict(handle, authorize=lambda: called.append(True))
	assert called == [True, True]
	assert store.bytes_used == 0


def test_retired_state_drains_existing_readers_and_closed_leases_release_references():
	store = ResidentTensors(16)
	handle = store.put(torch.arange(4.0))
	store.mark_published(handle)
	lease = store.acquire(handle)
	store.retire(handle)
	assert store.bytes_used == 16
	with pytest.raises(BufferInUse, match="retired"):
		store.acquire(handle)
	torch.testing.assert_close(lease.tensor, torch.arange(4.0))
	lease.close()
	assert store.bytes_used == 0


def test_retire_all_releases_failed_endpoint_buffers_after_last_lease():
	store = ResidentTensors(32)
	base = torch.arange(8.0)
	first = store.put(base[:4])
	store.put(base[4:])
	lease = store.acquire(first)
	store.retire_all()
	assert store.bytes_used == 32
	with pytest.raises(BufferInUse, match="retired"):
		store.acquire(first)
	lease.close()
	assert store.bytes_used == 0
	with pytest.raises(RuntimeError, match="closed"):
		_ = lease.tensor
	handle = store.put(torch.ones(4))
	store.mark_published(handle)
	store.retire(handle)
	assert store.bytes_used == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a visible GPU")
def test_cuda_result_readiness_tracks_the_producing_stream():
	store = ResidentTensors(4096 * 4)
	device = torch.device("cuda:0")
	stream = torch.cuda.Stream(device=device)
	with torch.cuda.stream(stream):
		value = torch.arange(4096, dtype=torch.float32, device=device) * 3
		handle = store.put(value)
	with store.acquire(handle, wait=True) as tensor:
		assert tensor.data_ptr() == value.data_ptr()
		assert store.is_ready(handle)
		torch.testing.assert_close(tensor.cpu(), torch.arange(4096, dtype=torch.float32) * 3)
