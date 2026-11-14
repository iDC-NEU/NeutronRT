"""Historical h recipes use old sums for changed rows, never new substitutes."""

from contextlib import ExitStack, contextmanager

import dgl
import pytest
import torch

from lib.graph.deltas import NeighborUpdate
from lib.graph.ownership import MissingState
from lib.models.aggregates import reconstruct_sage
from lib.runtime.aggregates.delta_state import DeltaState
from lib.runtime.aggregates.history import HistoricalAggregates
from lib.runtime.aggregates.tables import pack_tables
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from scripts.validate.inference import reference
from tests.runtime.test_prepared import _actors


@contextmanager
def _history(kind, *, missing=False):
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(gather=True))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
		features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
		features[0, 0] = 1e-20
		old_features, old_degrees = features.clone(), graph.in_degrees()
		model = workers[0].model
		expected = reference(model, graph, features, torch.arange(6))
		service = DistributedEmbeddingService(
			graph, features, (4, 4, 3), plane, retain_aggregates=True, aggregate_only=True, gather_inputs=True
		)
		service.ensure(torch.arange(6))
		for key in tuple(plane._shapes):
			if key.kind == "h":
				plane.retire(key)
		if missing:
			plane.retire(service.aggregate_catalog.selections(1, torch.tensor([0]), 0)[0].key)
		state = DeltaState(service, max_steps=2, anchor_bytes=4096)
		if kind == "features":
			state.prepare_features(torch.tensor([0, 3]))
			mutation = service._mutate_features(torch.tensor([0, 3]), torch.ones(2, 4))
		else:
			u, v = torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5])
			state.prepare_edges(u, v, delete=kind == "delete")
			mutation = service._mutate_edges(u, v, delete=kind == "delete")
		with state.updating():
			context = state.active
			service._begin_version(1, {}, pending_layers=(1, 2))
			for affected in service.affected.analyze(
				changed_features=mutation.changed_features, edge_targets=mutation.edge_targets
			):
				context.capture(affected.layer, affected.targets, 1)
				service._finish_analysis(affected.layer, affected.targets, 1)
				service.ensure(affected.targets, affected.layer)
			# Relocate all aggregate inputs before resolving their table locations.
			for key in tuple(plane._shapes):
				if key.kind == "a":
					location = plane.directory.locate(key)
					if location.owner != plane.storage.owner:
						plane.writeback(key, plane.storage.owner).result(5)
						plane.evict(key, location.owner)
				else:
					plane.retire(key)
			yield service, context, expected, old_features, old_degrees, model, workers[1].owner


@pytest.mark.parametrize("kind", ["features", "insert", "delete"])
@torch.no_grad()
def test_historical_prefixes_reconstruct_old_outputs_without_any_retained_h(kind):
	with _history(kind) as (service, context, expected, old_features, old_degrees, model, target):
		helper = HistoricalAggregates(service)
		nodes = torch.tensor([5, 0, 3, 1, 0])
		assert all(not layer.embeddings.snapshot.available.any() for layer in context.layers.values())
		helper.prepare(context, nodes, 2)
		tables = helper.tables(context, nodes, 2, target)
		pack_tables(tables)
		values = [tables[0].values]
		for table in tables[1:]:
			output = torch.empty(table.rows, table.width)
			for item in table.chunks:
				key = next(
					key
					for key in service.remote_plane._shapes
					if service.remote_plane.directory.locate(key, preferred=target) == item.location
				)
				with ExitStack() as leases:
					output[item.positions] = service.remote_plane.read_rows(key, item.rows, leases)
			values.append(output)
		degrees = helper.degrees(context, nodes, 2)
		torch.testing.assert_close(degrees, old_degrees[nodes], rtol=0, atol=0)
		torch.testing.assert_close(values[0], old_features[nodes], rtol=0, atol=0)
		actual = reconstruct_sage(
			model, values[0], [NeighborUpdate(value, degrees) for value in values[1:]], projected=True
		)
		torch.testing.assert_close(actual, expected[nodes], rtol=2e-6, atol=2e-6)
		assert not any(key.kind == "h" for key in service.remote_plane._shapes)
	with pytest.raises(MissingState, match="live graph-update"):
		helper.prepare(context, nodes, 2)


def test_missing_changed_old_sum_is_not_replaced_with_the_now_fresh_sum():
	with _history("features", missing=True) as (service, context, _expected, _features, _degrees, _model, target):
		helper = HistoricalAggregates(service)
		nodes = torch.tensor([0])
		assert service.aggregate_catalog.available(1, nodes, 1).all()
		with pytest.raises(MissingState, match="certified old aggregate"):
			helper.prepare(context, nodes, 2)
		with pytest.raises(MissingState, match="certified old aggregate"):
			helper.tables(context, nodes, 2, target)
