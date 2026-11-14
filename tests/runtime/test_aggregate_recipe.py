"""Full-neighbor tasks consume reconstructed inputs without retaining them."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.affected import AffectedPlanner
from lib.graph.ownership import StateKey
from lib.graph.serving import select_rows
from lib.runtime.aggregates.recipe import RecipeHeader, pack_recipe_compute
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk
from tests.runtime.test_prepared import _actors
from tests.runtime.test_reconstruction_worker import _state


@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("empty", [False, True])
@torch.inference_mode()
def test_recipe_consumes_only_features_and_sums_then_publishes_exact_h_and_sum(coalesced, gather, empty):
	graph = dgl.graph(([], []) if empty else ([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	task = AffectedPlanner(graph, 2).block(2, torch.tensor([5, 1, 0]))
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		outputs, sums = _state(plane, workers, graph, features)
		if gather:
			plane.writeback(sums[0][0], plane.storage.owner).result(5)
			plane.evict(sums[0][0], workers[0].owner)
		location = plane.directory.locate(sums[0][0], preferred=plane.storage.owner)
		tables = (
			InputTable(task.sources.numel(), 4, values=select_rows(features, task.sources)),
			InputTable(
				task.sources.numel(),
				4,
				chunks=(InputChunk(location, task.sources, torch.arange(task.sources.numel())),),
			),
		)
		target = workers[1]
		h = plane.directory.begin(StateKey(0, 2, 100), target.owner)
		a = plane.directory.begin(StateKey(0, 2, 200, "a"), target.owner)
		packet = pack_recipe_compute(task, h, a, (4, 4, 3), tables, graph.in_degrees(task.sources))
		assert RecipeHeader.decode(packet.header.tensor()) == packet.header
		assert packet.fields[4] is tables[0].values and packet.fields[-3] is task.sources
		before = target.store.bytes_used
		with ExitStack() as leases:

			def supply():
				supply_tables(plane, tables, target.owner, leases, gather=gather)

			results = transport.compute_from_aggregates(target.owner, packet, output_width=3, supply_inputs=supply)
			assert results == transport.compute_from_aggregates(
				target.owner, packet, output_width=3, supply_inputs=supply
			)
		assert target.store.bytes_used == before + task.targets.numel() * 3 * 4 * 2
		for attempt, result, expected in ((h, results[0], outputs[1]), (a, results[1], sums[1][2])):
			plane.publish_result(attempt, result)
			with ExitStack() as leases:
				torch.testing.assert_close(
					plane.read_rows(attempt.key, torch.arange(3), leases), expected[task.targets], atol=2e-6, rtol=2e-6
				)
		with ExitStack() as leases:
			torch.testing.assert_close(plane.read_rows(sums[0][0], torch.arange(6), leases), sums[0][2])


@torch.inference_mode()
def test_recipe_partial_budget_failure_abandons_both_outputs_and_can_retry():
	graph = dgl.graph(([0, 1], [1, 0]), num_nodes=6)
	features = torch.ones(6, 4)
	task = AffectedPlanner(graph, 2).block(2, torch.tensor([0, 1]))
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		_, sums = _state(plane, workers, graph, features)
		target = workers[1]
		tables = (
			InputTable(2, 4, values=features[task.sources]),
			InputTable(2, 4, chunks=(InputChunk(sums[0][1], task.sources, torch.arange(2)),)),
		)
		h = plane.directory.begin(StateKey(0, 2, 100), target.owner)
		a = plane.directory.begin(StateKey(0, 2, 200, "a"), target.owner)
		packet = pack_recipe_compute(task, h, a, (4, 4, 3), tables, graph.in_degrees(task.sources))
		before = target.store.bytes_used
		target.store.max_bytes = before + 24
		with ExitStack() as leases:

			def supply():
				supply_tables(plane, tables, target.owner, leases, gather=False)

			with pytest.raises(MemoryError):
				transport.compute_from_aggregates(target.owner, packet, output_width=3, supply_inputs=supply)
			assert target.store.bytes_used == before
			target.store.max_bytes = 4096
			results = transport.compute_from_aggregates(target.owner, packet, output_width=3, supply_inputs=supply)
			assert results[0].handle != results[1].handle
