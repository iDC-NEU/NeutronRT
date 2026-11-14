"""Real actor execution over FIFO channels; process transport is checked separately."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.ownership import StateKey
from lib.graph.recompute import RecomputePlanner
from lib.graph.serving import select_rows
from lib.runtime.aggregates.protocol import pack_aggregate_compute
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk, pack_compute
from lib.runtime.transport import ModelExecutionError
from tests.runtime.test_prepared import _actors


def _packet(plane, worker, task, features, *, chunk=1, inputs=()):
	h = plane.directory.begin(StateKey(0, task.layer, chunk), worker.owner)
	a = plane.directory.begin(StateKey(0, task.layer, 100 + chunk, "a"), worker.owner)
	packet = pack_aggregate_compute(task, h, a, width=4, features=features, inputs=inputs)
	return packet, h, a


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("peer", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@torch.inference_mode()
def test_paired_outputs_replay_peer_inputs_and_aggregate_writeback(coalesced, peer, empty):
	graph = dgl.graph(([], []) if empty else ([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	planner = RecomputePlanner(graph, 2)
	plan = planner.plan(torch.arange(6), 2, 0)
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		keys = []
		previous = None
		current = features
		for task in plan.tasks:
			worker = workers[task.layer - 1] if peer else workers[0]
			inputs = (
				() if task.layer == 1 else (InputChunk(previous, task.sources, torch.arange(task.sources.numel())),)
			)
			packet, h, a = _packet(
				plane,
				worker,
				task,
				select_rows(features, task.sources) if task.layer == 1 else None,
				chunk=task.layer,
				inputs=inputs,
			)
			with ExitStack() as leases:

				def supply(inputs=inputs, worker=worker, leases=leases):
					plane.supply_inputs(inputs, worker.owner, 4, leases)

				results = transport.compute_with_aggregate(
					worker.owner, packet, output_width=4 if task.layer == 1 else 3, supply_inputs=supply
				)
				assert results == transport.compute_with_aggregate(
					worker.owner, packet, output_width=4 if task.layer == 1 else 3, supply_inputs=supply
				)
			previous = plane.publish_result(h, results[0])
			plane.publish_result(a, results[1])
			keys.extend((h.key, a.key))
			projected = workers[0].model.layers[task.layer - 1].fc_neigh(current)
			expected_sums = torch.zeros_like(projected)
			u, v = graph.edges()
			expected_sums.index_add_(0, v, projected[u])
			current = workers[0].model(task.layer - 1, graph, current)
			rows = torch.tensor([5, 0, 2, 0])
			with ExitStack() as leases:
				torch.testing.assert_close(plane.read_rows(h.key, rows, leases), current[rows], atol=2e-6, rtol=2e-6)
				torch.testing.assert_close(plane.read_rows(a.key, rows, leases), expected_sums[rows])
			plane.writeback(a.key, plane.storage.owner).result(5)
			plane.evict(a.key, worker.owner)
			with ExitStack() as leases:
				# The h read also fences earlier retirement on the actor's control FIFO.
				plane.read_rows(h.key, rows, leases)
				torch.testing.assert_close(plane.read_rows(a.key, rows, leases), expected_sums[rows])
			assert not worker._aggregate_tasks.owns_request(h.number)
		for key in keys:
			plane.retire(key)
	assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["budget", "model"])
@torch.inference_mode()
def test_failed_pair_cleans_partial_state_and_drains_both_reply_slots(monkeypatch, failure):
	graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
	features = torch.ones(2, 4)
	task = RecomputePlanner(graph, 1).plan(torch.arange(2), 1, 0).tasks[0]
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		worker = workers[0]
		packet, _, _ = _packet(plane, worker, task, select_rows(features, task.sources))
		if failure == "budget":
			worker.store.max_bytes = 32  # Only one of the two 2x4 float buffers fits.
		else:
			original = worker._apply_with_aggregate
			monkeypatch.setattr(
				worker, "_apply_with_aggregate", lambda *_args: (_ for _ in ()).throw(ValueError("model"))
			)
		with pytest.raises(MemoryError if failure == "budget" else ModelExecutionError):
			transport.compute_with_aggregate(worker.owner, packet, output_width=4, supply_inputs=lambda: None)
		assert worker.store.bytes_used == 0
		assert not worker._aggregate_tasks.owns_request(packet.header.request)
		worker.store.max_bytes = 4096
		if failure == "model":
			monkeypatch.setattr(worker, "_apply_with_aggregate", original)
		packet, h, _ = _packet(plane, worker, task, select_rows(features, task.sources), chunk=2)
		results = transport.compute_with_aggregate(worker.owner, packet, output_width=4, supply_inputs=lambda: None)
		assert results[0].handle != results[1].handle and worker.store.bytes_used == 64
		# A paired request cannot be replayed through the ordinary compute opcode.
		normal = pack_compute(task, h, width=4, features=select_rows(features, task.sources))
		with pytest.raises(ModelExecutionError):
			transport.compute(worker.owner, normal, output_width=4, supply_inputs=lambda: None)
