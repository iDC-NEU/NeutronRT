"""Sparse worker commands match full inference without constructing blocks."""

from contextlib import ExitStack
from dataclasses import replace

import dgl
import pytest
import torch

from lib.graph.delta_plan import compile_delta_routes
from lib.graph.ownership import StateKey
from lib.runtime.aggregates.delta import pack_delta_compute
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.protocol import InputChunk
from lib.runtime.transport import ModelExecutionError, ReadyResult
from tests.runtime.test_prepared import _actors
from tests.runtime.test_reconstruction_worker import _state


def _table(plane, key, nodes, width, target):
	return InputTable(
		nodes.numel(),
		width,
		chunks=(InputChunk(plane.directory.locate(key, preferred=target), nodes, torch.arange(nodes.numel())),)
		if nodes.numel()
		else (),
	)


@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_sparse_worker_propagates_two_layers_with_peer_and_storage_anchors(coalesced, gather):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	old_features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	features = old_features.clone()
	features[[0, 3]] += 1.25
	old_degrees = graph.in_degrees()
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		old_h, sums = _state(plane, workers, graph, old_features)
		owner = workers[0]
		handle = owner.store.put(old_h[0])
		owner.store.mark_published(handle)
		old_h_key = StateKey(0, 1, 900)
		attempt = plane.directory.begin(old_h_key, owner.owner)
		plane.publish_result(attempt, ReadyResult(handle, 0, 1, 900, (6, 4), owner.store.bytes_used))
		if gather:
			for key, location, _ in sums:
				plane.writeback(key, plane.storage.owner).result(5)
				plane.evict(key, location.owner)
		_, _, ids = graph.edge_ids(torch.tensor([0]), torch.tensor([1]), return_uv=True)
		graph.remove_edges(ids)
		graph.add_edges(torch.tensor([3, 3]), torch.tensor([2, 2]))
		edge_sources, edge_targets, weights = torch.tensor([0, 3]), torch.tensor([1, 2]), torch.tensor([-2, 2])
		changed = torch.tensor([0, 3])
		current = features
		previous = previous_nodes = None
		for layer in (1, 2):
			plan = compile_delta_routes(graph, changed, edge_sources, edge_targets, weights, max_edges=64)
			target = workers[2 - layer].owner
			old_sum = _table(plane, sums[layer - 1][0], plan.targets, 4 if layer == 1 else 3, target)
			if layer == 1:
				self_new = InputTable(plan.targets.numel(), 4, values=features[plan.targets])
				new_input = InputTable(plan.inputs.value_nodes.numel(), 4, values=features[plan.inputs.value_nodes])
				old_input = InputTable(plan.inputs.value_nodes.numel(), 4, values=old_features[plan.inputs.value_nodes])
				structural = InputTable(plan.edges.value_nodes.numel(), 4, values=old_features[plan.edges.value_nodes])
			else:
				self_new = _table(plane, previous, torch.searchsorted(previous_nodes, plan.targets), 4, target)
				new_input = _table(
					plane, previous, torch.searchsorted(previous_nodes, plan.inputs.value_nodes), 4, target
				)
				old_input = _table(plane, old_h_key, plan.inputs.value_nodes, 4, target)
				structural = _table(plane, old_h_key, plan.edges.value_nodes, 4, target)
			tables = (old_sum, self_new, new_input, old_input, structural)
			h = plane.directory.begin(StateKey(1, layer, 100 + layer), target)
			a = plane.directory.begin(StateKey(1, layer, 200 + layer, "a"), target)
			packet = pack_delta_compute(plan, h, a, (4, 4, 3), tables, old_degrees[plan.targets])
			with ExitStack() as inputs:

				def supply(tables=tables, target=target, inputs=inputs):
					supply_tables(plane, tables, target, inputs, gather=gather)

				results = transport.compute_delta(
					target, packet, output_width=4 if layer == 1 else 3, supply_inputs=supply
				)
				assert results == transport.compute_delta(
					target, packet, output_width=4 if layer == 1 else 3, supply_inputs=supply
				)
			plane.publish_result(h, results[0])
			plane.publish_result(a, results[1])
			projected = workers[0].model.layers[layer - 1].fc_neigh(current)
			expected_sum = torch.zeros_like(projected)
			u, v = graph.edges()
			expected_sum.index_add_(0, v, projected[u])
			current = workers[0].model(layer - 1, graph, current)
			with ExitStack() as leases:
				torch.testing.assert_close(
					plane.read_rows(h.key, torch.arange(plan.targets.numel()), leases),
					current[plan.targets],
					atol=3e-6,
					rtol=3e-6,
				)
				torch.testing.assert_close(
					plane.read_rows(a.key, torch.arange(plan.targets.numel()), leases),
					expected_sum[plan.targets],
					atol=3e-6,
					rtol=3e-6,
				)
				for key, _, values in sums:
					torch.testing.assert_close(plane.read_rows(key, torch.arange(6), leases), values)
			changed, previous, previous_nodes = plan.targets, h.key, plan.targets
		assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", ["degree", "budget"])
@torch.inference_mode()
def test_sparse_failure_preserves_old_sums_and_allows_retry(failure):
	from tests.runtime.test_delta_protocol import _packet_inputs

	plan, _, _, tables, degrees = _packet_inputs()
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=4096)
		stack.callback(plane.close)
		worker = workers[0]
		h = plane.directory.begin(StateKey(1, 2, 100), worker.owner)
		a = plane.directory.begin(StateKey(1, 2, 200, "a"), worker.owner)
		original = tables[0].values.clone()
		bad_plan = (
			replace(plan, edges=replace(plan.edges, weights=torch.tensor([-99]))) if failure == "degree" else plan
		)
		packet = pack_delta_compute(bad_plan, h, a, (4, 4, 3), tables, degrees)
		if failure == "budget":
			worker.store.max_bytes = 24
		with pytest.raises(ModelExecutionError if failure == "degree" else MemoryError):
			transport.compute_delta(worker.owner, packet, output_width=3, supply_inputs=lambda: None)
		assert worker.store.bytes_used == 0
		torch.testing.assert_close(tables[0].values, original)
		worker.store.max_bytes = 4096
		packet = pack_delta_compute(plan, h, a, (4, 4, 3), tables, degrees)
		results = transport.compute_delta(worker.owner, packet, output_width=3, supply_inputs=lambda: None)
		assert results[0].handle != results[1].handle
