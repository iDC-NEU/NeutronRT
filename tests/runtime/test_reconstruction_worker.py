"""Row recipes exercise the real actor, ownership and peer-input paths."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.ownership import StateKey
from lib.runtime.aggregates.reconstruction import pack_reconstruction
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk
from lib.runtime.transport import ModelExecutionError, ReadyResult
from tests.runtime.test_prepared import _actors


def _state(plane, workers, graph, features):
	model = workers[0].model
	current, outputs, sums = features, [], []
	u, v = graph.edges()
	for layer in range(2):
		projected = model.layers[layer].fc_neigh(current)
		aggregate = projected.new_zeros(projected.shape)
		aggregate.index_add_(0, v, projected[u])
		current = model(layer, graph, current)
		outputs.append(current)
		worker = workers[layer]
		handle = worker.store.put(aggregate)
		worker.store.mark_published(handle)
		attempt = plane.directory.begin(StateKey(0, layer + 1, layer + 1, "a"), worker.owner)
		location = plane.publish_result(
			attempt, ReadyResult(handle, 0, layer + 1, layer + 1, tuple(aggregate.shape), worker.store.bytes_used)
		)
		sums.append((attempt.key, location, aggregate))
	return outputs, sums


def _tables(plane, workers, features, outputs, sums, rows, start):
	tables = []
	if start:
		worker = workers[0]
		handle = worker.store.put(outputs[0])
		worker.store.mark_published(handle)
		attempt = plane.directory.begin(StateKey(0, 1, 100), worker.owner)
		location = plane.publish_result(
			attempt, ReadyResult(handle, 0, 1, 100, tuple(outputs[0].shape), worker.store.bytes_used)
		)
		tables.append(
			InputTable(
				rows.numel(),
				4,
				chunks=(InputChunk(location, rows, torch.arange(rows.numel())),) if rows.numel() else (),
			)
		)
	else:
		tables.append(InputTable(rows.numel(), 4, values=features[rows]))
	for key, _location, aggregate in sums[start:]:
		location = plane.directory.locate(key, preferred=plane.storage.owner)
		chunks = tuple(
			InputChunk(location, selected, positions)
			for selected, positions in zip(rows.split(2), torch.arange(rows.numel()).split(2), strict=True)
			if selected.numel()
		)
		tables.append(InputTable(rows.numel(), aggregate.shape[1], chunks=chunks))
	return tuple(tables)


@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("empty", [False, True])
@torch.inference_mode()
def test_reconstructs_selected_rows_from_local_peer_and_storage_sums(coalesced, gather, start, empty):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	rows = torch.empty(0, dtype=torch.int64) if empty else torch.tensor([5, 1, 0, 1])
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		outputs, sums = _state(plane, workers, graph, features)
		if gather:
			plane.writeback(sums[0][0], plane.storage.owner).result(5)
			plane.evict(sums[0][0], workers[0].owner)
		tables = _tables(plane, workers, features, outputs, sums, rows, start)
		target = workers[1]
		attempt = plane.directory.begin(StateKey(0, 2, 200), target.owner)
		packet = pack_reconstruction(attempt, (4, 4, 3), tables, graph.in_degrees(rows), start_layer=start)
		before = target.store.bytes_used
		with ExitStack() as leases:

			def supply():
				supply_tables(plane, tables, target.owner, leases, gather=gather)

			result = transport.reconstruct(target.owner, packet, output_width=3, supply_inputs=supply)
			assert result == transport.reconstruct(target.owner, packet, output_width=3, supply_inputs=supply)
		assert result.bytes_used == before + rows.numel() * 3 * 4
		plane.publish_result(attempt, result)
		with ExitStack() as leases:
			torch.testing.assert_close(
				plane.read_rows(attempt.key, torch.arange(rows.numel()), leases), outputs[1][rows], atol=2e-6, rtol=2e-6
			)
			for key, _, aggregate in sums:
				torch.testing.assert_close(plane.read_rows(key, torch.arange(6), leases), aggregate)
		plane.retire(attempt.key)
	assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["budget", "model"])
@torch.inference_mode()
def test_failed_reconstruction_preserves_sums_and_can_retry_after_error(monkeypatch, failure):
	graph = dgl.graph(([0, 1], [1, 0]), num_nodes=6)
	features = torch.ones(6, 4)
	rows = torch.tensor([1, 0])
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		outputs, sums = _state(plane, workers, graph, features)
		tables = _tables(plane, workers, features, outputs, sums, rows, 0)
		target = workers[1]
		attempt = plane.directory.begin(StateKey(0, 2, 200), target.owner)
		packet = pack_reconstruction(attempt, (4, 4, 3), tables, graph.in_degrees(rows))
		before, apply = target.store.bytes_used, target._reconstruct_sage
		if failure == "budget":
			target.store.max_bytes = before
		else:
			monkeypatch.setattr(target, "_reconstruct_sage", lambda *_args: (_ for _ in ()).throw(ValueError("model")))
		with ExitStack() as leases:

			def supply():
				supply_tables(plane, tables, target.owner, leases, gather=False)

			with pytest.raises(MemoryError if failure == "budget" else ModelExecutionError):
				transport.reconstruct(target.owner, packet, output_width=3, supply_inputs=supply)
			assert target.store.bytes_used == before and attempt.number not in target._attempts
			target.store.max_bytes = 4096
			monkeypatch.setattr(target, "_reconstruct_sage", apply)
			result = transport.reconstruct(target.owner, packet, output_width=3, supply_inputs=supply)
		plane.publish_result(attempt, result)
		with ExitStack() as leases:
			torch.testing.assert_close(plane.read_rows(attempt.key, torch.arange(2), leases), outputs[1][rows])
