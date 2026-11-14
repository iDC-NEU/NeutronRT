from contextlib import ExitStack

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import StateKey
from lib.runtime.aggregates.tables import InputTable, pack_tables, receive_tables, resolve_tables, supply_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk
from lib.runtime.transport import ReadyResult
from tests.runtime.test_prepared import _actors


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("gather", [False, True])
def test_mixed_width_tables_resolve_root_local_and_peer_values_without_model_width_assumptions(coalesced, gather):
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)

		def put(worker, values, chunk):
			handle = worker.store.put(values)
			worker.store.mark_published(handle)
			attempt = plane.directory.begin(StateKey(0, 1, chunk, "a"), worker.owner)
			return plane.publish_result(
				attempt, ReadyResult(handle, 0, 1, chunk, tuple(values.shape), worker.store.bytes_used)
			)

		first = put(workers[0], torch.arange(1, 9, dtype=torch.float32).reshape(4, 2), 1)
		second = put(workers[0], torch.tensor([[9.0, 10.0]]), 2)
		local = put(workers[1], torch.tensor([[11.0, 12.0, 13.0]]), 3)
		remote = put(workers[0], torch.tensor([[14.0, 15.0, 16.0]]), 4)
		features = torch.arange(8, dtype=torch.float32).reshape(2, 4)
		tables = (
			InputTable(2, 4, values=features),
			InputTable(
				3,
				2,
				chunks=(
					InputChunk(first, torch.tensor([3, 0]), torch.tensor([2, 0])),
					InputChunk(second, torch.tensor([0]), torch.tensor([1])),
				),
			),
			InputTable(
				2,
				3,
				chunks=(
					InputChunk(local, torch.tensor([0]), torch.tensor([1])),
					InputChunk(remote, torch.tensor([0]), torch.tensor([0])),
				),
			),
		)
		packet = pack_tables(tables)
		tail = torch.tensor([17, 29], dtype=torch.int64)
		layout = PackedLayout(tuple(TensorSpec.of(value) for value in (*packet.fields, tail)))
		if coalesced:
			transport.comm.isend_tensors((*packet.fields, tail), 2).wait()
		else:
			for value in (*packet.fields, tail):
				if value.numel():
					transport.comm.isend(value, 2).wait()
		fields, extra = receive_tables(workers[1], layout.nbytes, 3, extra_specs=(TensorSpec.of(tail),))
		assert torch.equal(extra[0], tail)
		with ExitStack() as leases:
			supply_tables(plane, tables, workers[1].owner, leases, gather=gather)
			values = resolve_tables(workers[1], fields, leases)
			torch.testing.assert_close(values[0], features)
			torch.testing.assert_close(values[1], torch.tensor([[1.0, 2.0], [9.0, 10.0], [7.0, 8.0]]))
			torch.testing.assert_close(values[2], torch.tensor([[14.0, 15.0, 16.0], [11.0, 12.0, 13.0]]))
	assert not torch.cuda.is_initialized()
