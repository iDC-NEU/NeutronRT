from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

from lib.graph.catalog import RowSelection
from lib.graph.ownership import Location, Owner, StateKey
from lib.models.spec import ModelSpec
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import READ_MANY, ComputeHeader
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.runtime.worker import OwnedWorker


class _Comm:
	device = torch.device("cpu")

	def __init__(self, rank, received=()):
		self.rank = rank
		self.received = list(received)
		self.sent = []
		self.receive_count = 0
		self.complete = False

	def recv_tensor(self, shape, **_kwargs):
		self.receive_count += 1
		value = self.received.pop(0)
		assert tuple(value.shape) == ((shape,) if isinstance(shape, int) else tuple(shape))
		return value

	def empty(self, shape, dtype=torch.float32):
		return torch.empty(shape, dtype=dtype)

	def isend(self, value, peer, **_kwargs):
		self.sent.append((value, peer))
		return SimpleNamespace(wait=lambda: None, is_completed=lambda: self.complete)

	def send(self, value, peer, **kwargs):
		self.isend(value, peer, **kwargs).wait()

	def wait_all(self, requests):
		for request in requests:
			request.wait()


def _worker(comm, *, gather=False):
	return OwnedWorker(
		comm,
		ModelSpec("sage", 2, 2, 3).build(),
		Owner(comm.rank, 1),
		max_bytes=4096,
		control_epoch=9,
		gather_inputs=gather,
	)


def test_grouped_source_reads_gather_once_and_retain_all_source_leases_until_send_finishes():
	comm = _Comm(1)
	worker = _worker(comm)
	first = worker.store.put(torch.tensor([[10.0, 11], [12, 13], [14, 15]]))
	second = worker.store.put(torch.tensor([[20.0, 21], [22, 23]]))
	worker.store.mark_published(first)
	worker.store.mark_published(second)
	comm.received = [torch.tensor([[first, 2], [second, 1]]), torch.tensor([2, 0, 1])]
	worker.read_many(torch.tensor([READ_MANY, 0, 1, 2, 2, 3, 2, 0]))
	assert len(comm.sent) == 1 and comm.sent[0][1] == 2
	torch.testing.assert_close(comm.sent[0][0], torch.tensor([[14.0, 15], [10, 11], [22, 23]]))
	worker.store.retire(first)
	worker.store.retire(second)
	assert worker.store.bytes_used == 40
	comm.complete = True
	worker._drain()
	assert worker.store.bytes_used == 0


@pytest.mark.parametrize("gather", [False, True])
def test_mixed_local_and_interleaved_remote_groups_preserve_positions_and_duplicates(gather):
	remote = torch.tensor([[20.0, 21], [22, 23], [24, 25], [26, 27]])
	comm = _Comm(1, [remote] if gather else [remote[:1], remote[1:]])
	worker = _worker(comm, gather=gather)
	handle = worker.store.put(torch.tensor([[10.0, 11], [12, 13]]))
	worker.store.mark_published(handle)
	header = ComputeHeader(1, 1, 2, 6, 3, 0, 3)
	fields = [torch.empty(0)] * 10
	fields[6] = torch.tensor([[2, 1, 3, 1], [1, 1, handle, 2], [2, 1, 4, 3]])
	fields[7] = torch.tensor([9, 1, 0, 2, 0, 2])
	fields[8] = torch.tensor([5, 1, 3, 0, 2, 4])
	with ExitStack() as stack:
		actual = worker._inputs(header, tuple(fields), stack)
		torch.testing.assert_close(actual, torch.tensor([[22.0, 23], [12, 13], [24, 25], [10, 11], [26, 27], [20, 21]]))
	assert comm.receive_count == (1 if gather else 2)


def test_grouped_transport_keeps_integer_metadata_and_rejects_cross_generation_sources():
	comm = _Comm(0)
	source, target = Owner(1, 2), Owner(2, 3)
	transport = OwnedTransport(comm, 9, [RemoteEndpoint(owner, comm.device, 4096) for owner in (source, target)])
	requests = [(Location(source, 2**55 + 1), torch.tensor([4, 2])), (Location(source, 3), torch.tensor([9]))]
	transport.fetch_peer_many(requests, 8, target)
	assert comm.sent[0][0].tolist() == [READ_MANY, 0, 2, 2, 2, 3, 8, 0]
	assert comm.sent[1][0].tolist() == [[2**55 + 1, 2], [3, 1]]
	assert comm.sent[2][0].tolist() == [4, 2, 9]
	with pytest.raises(ValueError, match="generation"):
		transport.read_many([requests[0], (Location(Owner(1, 3), 3), torch.tensor([1]))], 8)


def test_root_gathers_cpu_and_remote_chunks_once_per_owner_and_restores_query_order():
	comm = _Comm(0)
	owner = Owner(1, 1)
	transport = OwnedTransport(comm, 9, [RemoteEndpoint(owner, comm.device, 4096)])
	plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
	stored = plane.storage.store.put(torch.tensor([[10.0, 11], [12, 13], [14, 15]]))
	plane.storage.store.mark_published(stored)
	keys = [StateKey(0, 1, index) for index in (1, 2, 3)]
	for key, source, handle in zip(keys, (owner, plane.storage.owner, owner), (3, stored, 7), strict=True):
		plane.directory.publish(plane.directory.begin(key, source), handle)
	selections = [
		RowSelection(keys[0], torch.tensor([3]), torch.tensor([4])),
		RowSelection(keys[1], torch.tensor([1, 4]), torch.tensor([2, 0])),
		RowSelection(keys[2], torch.tensor([0, 2]), torch.tensor([1, 1])),
	]
	calls = []

	def read_many(requests, width):
		calls.append((requests, width))
		return SimpleNamespace(result=lambda: torch.tensor([[40.0, 41], [70, 71], [70, 71]]))

	transport.read_many = read_many
	try:
		with ExitStack() as stack:
			actual = plane.gather_rows(selections, 2, stack)
			torch.testing.assert_close(actual, torch.tensor([[70.0, 71], [14, 15], [70, 71], [40, 41], [10, 11]]))
		assert len(calls) == 1 and len(calls[0][0]) == 2
	finally:
		plane.close()
