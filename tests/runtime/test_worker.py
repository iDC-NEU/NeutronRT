from datetime import timedelta

import dgl
import pytest
import torch
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager
from lib.graph.ownership import EmbeddingDirectory, Location, Owner, StateKey
from lib.graph.recompute import RecomputePlanner
from lib.graph.serving import select_rows
from lib.models import GNNModel
from lib.runtime.protocol import READ, READY, RETIRE, STOP, InputChunk, pack_compute
from lib.runtime.worker import REPLY_FIELDS, OwnedWorker
from tests.dist.test_transport import _join_processes


def _owned(rank, rendezvous, name, use_cuda, peer_inputs):
	torch.set_num_threads(1)
	devices = ("cpu", "cuda:0", "cuda:1") if use_cuda else ("cpu", "cpu", "cpu")
	manager = CommManager.initialize(
		3, rank, device=devices[rank], init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	control = manager.create_epoch([0, 1, 2], timeout=timedelta(seconds=20))
	torch.manual_seed(19)
	model = GNNModel(name, 4, 4, 3, 2, aggregator_type="mean", num_heads=2).eval().to(devices[rank])
	for layer in model.layers:
		if hasattr(layer, "set_allow_zero_in_degree"):
			layer.set_allow_zero_in_degree(True)
	if rank:
		OwnedWorker(manager, model, Owner(rank, 1), max_bytes=96 if peer_inputs else 132, control_epoch=control).run()
		return
	try:
		assert not torch.cuda.is_initialized(), "The storage coordinator must remain CPU-only"
		graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4], [0, 1, 2, 2, 3, 4, 3]), num_nodes=6)
		features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
		plan = RecomputePlanner(graph, 2, source_degree_sensitive=name == "gcn").plan(torch.tensor([3, 0, 5, 3]), 2, 0)
		directory = EmbeddingDirectory()
		owners = {rank: directory.register_owner(rank) for rank in (1, 2)}
		states = {}
		packets = {}
		first_reply = None
		for task in plan.tasks:
			owner = owners[task.layer if peer_inputs else 1]
			key = StateKey(0, task.layer, task.layer)
			attempt = directory.begin(key, owner)
			inputs = []
			root = None
			if task.layer == 1:
				root = select_rows(features, task.sources)
			else:
				previous, location = states[task.layer - 1]
				mapping = torch.full((6,), -1, dtype=torch.int64)
				mapping[previous.targets] = torch.arange(previous.targets.numel())
				inputs = [InputChunk(location, mapping[task.sources], torch.arange(task.sources.numel()))]
			packet = pack_compute(task, attempt, width=4, inputs=inputs, features=root)
			packets[task.layer] = packet
			manager.send(packet.header.tensor(), owner.rank, epoch=control)
			manager.isend_tensors(packet.fields, owner.rank).wait()
			reply = torch.empty(REPLY_FIELDS, dtype=torch.int64)
			ready = manager.irecv(reply, owner.rank, epoch=control)
			for item in inputs:
				if item.location.owner == owner:
					continue
				manager.send(
					torch.tensor(
						[
							READ,
							attempt.number,
							item.location.owner.generation,
							item.location.handle,
							owner.rank,
							item.rows.numel(),
							4,
							0,
						]
					),
					item.location.owner.rank,
					epoch=control,
				)
				manager.send(item.rows, item.location.owner.rank)
			ready.wait()
			assert reply[:3].tolist() == [READY, attempt.number, owner.generation]
			assert reply[4:7].tolist() == [key.version, key.layer, key.chunk]
			assert reply[7:9].tolist() == [task.targets.numel(), 4 if task.layer == 1 else 3]
			assert directory.publish(attempt, int(reply[3]))
			states[task.layer] = task, Location(owner, int(reply[3]))
			if task.layer == 1:
				first_reply = reply.clone()
		first_owner_bytes = first_reply[9] if peer_inputs else reply[9]
		assert not torch.cuda.is_initialized()
		# Leave an explicit CPU writeback unmatched on the data group. A new
		# completion must still arrive on the separate control epoch.
		first_task, first_location = states[1]
		manager.send(
			torch.tensor(
				[
					READ,
					100,
					1,
					first_location.handle,
					0,
					first_task.targets.numel(),
					4,
					0,
				]
			),
			1,
			epoch=control,
		)
		manager.send(torch.arange(first_task.targets.numel()), 1)
		packet = packets[1]
		manager.send(packet.header.tensor(), 1, epoch=control)
		manager.isend_tensors(packet.fields, 1).wait()
		duplicate = manager.recv_tensor(REPLY_FIELDS, src=1, dtype=torch.int64, epoch=control)
		assert torch.equal(duplicate[:9], first_reply[:9]), "Retry changed the handle or waited for writeback"
		assert duplicate[9] == first_owner_bytes
		with torch.inference_mode():
			intermediate = model(0, graph, features)
			full = model(1, graph, intermediate)
		writeback = manager.recv_tensor((first_task.targets.numel(), 4), src=1)
		torch.testing.assert_close(writeback, intermediate[first_task.targets], rtol=2e-5, atol=2e-5)
		last_task, last_location = states[2]
		rows = torch.tensor([2, 0, 2])
		manager.send(
			torch.tensor([READ, 101, 1, last_location.handle, 0, 3, 3, 0]), last_location.owner.rank, epoch=control
		)
		manager.send(rows, last_location.owner.rank)
		actual = manager.recv_tensor((3, 3), src=last_location.owner.rank)
		torch.testing.assert_close(actual, full[last_task.targets[rows]], rtol=2e-5, atol=2e-5)
		# Retirement drains outstanding readers before the bounded store can
		# publish a replacement chunk. The request identity remains monotonic.
		manager.send(torch.tensor([RETIRE, 0, 1, first_location.handle, 0, 0, 0, 0]), 1, epoch=control)
		attempt = directory.begin(StateKey(0, 1, 3), owners[1])
		packet = pack_compute(first_task, attempt, width=4, features=select_rows(features, first_task.sources))
		manager.send(packet.header.tensor(), 1, epoch=control)
		manager.isend_tensors(packet.fields, 1).wait()
		replaced = manager.recv_tensor(REPLY_FIELDS, src=1, dtype=torch.int64, epoch=control)
		assert replaced[0] == READY
		assert replaced[3] != first_location.handle
		assert replaced[9] == first_owner_bytes
		assert not torch.cuda.is_initialized()
	finally:
		for peer in (1, 2):
			manager.send(torch.tensor([STOP, 0, 1, 0, 0, 0, 0, 0]), peer, epoch=control)
		manager.destroy()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize("peer_inputs", [False, True])
def test_process_workers_publish_handles_fetch_peers_and_defer_writeback(tmp_path, name, use_cuda, peer_inputs):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_owned, args=(str(tmp_path / "store"), name, use_cuda, peer_inputs), nprocs=3, join=False)
	_join_processes(context, timeout=90)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_contiguous_gpu_row_metadata_preserves_the_embedding_view():
	values = torch.arange(24, device="cuda:0").reshape(6, 4)
	selected = select_rows(values, torch.tensor([1, 2, 3], device="cuda:0"))
	assert selected.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
	torch.testing.assert_close(selected, values[1:4])
