from contextlib import ExitStack, contextmanager

import dgl
import pytest
import torch

from lib.graph.catalog import RowSelection
from lib.graph.ownership import MissingState
from lib.runtime.aggregates.delta_state import DeltaState
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from tests.runtime.test_prepared import _actors


@contextmanager
def _state(*, budget=4096, row_refresh=False, aggregate_only=False):
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		graph = dgl.graph(([0, 0, 1], [1, 1, 2]), num_nodes=6)
		service = DistributedEmbeddingService(
			graph, torch.ones(6, 4), (4, 4, 3), plane, retain_aggregates=True, aggregate_only=aggregate_only
		)
		service.ensure(torch.arange(6))
		yield DeltaState(service, max_steps=2, anchor_bytes=budget, row_refresh=row_refresh), workers


def test_aggregate_only_context_retains_sums_but_only_integer_frontiers_for_old_h():
	with _state(aggregate_only=True) as (state, _workers):
		service, nodes = state.service, torch.tensor([0, 1])
		state.prepare_features(nodes)
		with state.updating():
			service._begin_version(1, {}, pending_layers=(1, 2))
			context = state.active
			context.capture(1, nodes, 1)
			h, a = context.layers[1].embeddings.snapshot, context.layers[1].aggregates.snapshot
			assert h.nodes.data_ptr() == nodes.data_ptr() and not h.available.any() and not h.selections
			assert a.available.all()
			assert all(key.kind == "a" for key in service.remote_plane._retained_states)
			assert context.anchor_bytes == 2 * 24 + 6 * 4 * 4
			for key in tuple(service.remote_plane._shapes):
				if key.kind == "h":
					service.remote_plane.retire(key)
			assert context.enabled and not service.remote_plane._pending_retire
		assert not service.remote_plane._retained_states


def test_context_retains_old_buffers_until_failure_then_releases_obsolete_keys():
	with _state() as (state, _workers):
		service, nodes = state.service, torch.tensor([0, 1])
		state.prepare_features(nodes)
		assert state.pending.features.values.data_ptr() != service.features.data_ptr()
		with pytest.raises(RuntimeError, match="interrupted"), state.updating():
			context = state.active
			assert not context.enabled
			service._begin_version(1, {}, pending_layers=(1, 2))
			context.capture(1, nodes, 1)
			anchors = context.layers[1]
			assert anchors.embeddings.snapshot.nodes.data_ptr() == nodes.data_ptr()
			keys = [g.key for a in (anchors.embeddings, anchors.aggregates) for g in a.snapshot.selections]
			assert context.anchor_bytes == 2 * 24 + 2 * 6 * 4 * 4
			for key in keys:
				service.remote_plane.retire(key)
				service.remote_plane.directory.locate(key)
			assert service.remote_plane._retained_states
			raise RuntimeError("interrupted")
		assert state.active is None and state.pending is None
		assert not service.remote_plane._retained_states and not service.remote_plane._pending_retire
		for key in keys:
			with pytest.raises(MissingState):
				service.remote_plane.directory.locate(key)


def test_anchor_budget_counts_full_backings_and_releases_rejected_acquisitions():
	with _state(budget=200) as (state, _workers):
		state.prepare_features(torch.tensor([0]))
		with state.updating():
			service, context = state.service, state.active
			service._begin_version(1, {}, pending_layers=(1, 2))
			context.capture(1, torch.tensor([0]), 1)
			assert context.disabled and not context.layers
			assert not service.remote_plane._retained_states
	with _state(budget=0) as (state, _workers):
		state.prepare_features(torch.tensor([0]))
		assert not state.pending.enabled
		with state.updating():
			assert state.active.disabled


def test_actual_edge_multiplicity_and_owner_change_disable_old_transition():
	with _state() as (state, _workers):
		state.prepare_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 4]), delete=True)
		assert state.pending.edges.weights.tolist() == [-2]
		with state.updating():
			service, context = state.service, state.active
			service._begin_version(1, {}, pending_layers=(1, 2))
			context.capture(1, torch.tensor([1]), 1)
			assert context.enabled
			with pytest.raises(ValueError, match="once"):
				context.capture(1, torch.tensor([1]), 1)
			service.remote_plane.remotes.pop(next(iter(service.remote_plane.remotes)))
			assert not context.enabled


def test_refresh_depth_follows_current_aggregate_keys_and_prunes_retired_history():
	with _state() as (state, _workers):
		service, nodes = state.service, torch.arange(6)
		key = service.aggregate_catalog.selections(1, nodes, 0)[0].key
		state.record(key, 2)
		assert state.depth([key]) == 2
		state.prepare_features(nodes)
		with state.updating():
			service._begin_version(1, {}, pending_layers=(1, 2))
			state.active.capture(1, nodes, 1)
			service._finish_analysis(1, nodes, 1)
			service.ensure(nodes, 1)
			assert state.depth([key]) == 2
		assert key not in state._depths


@pytest.mark.parametrize("steps,budget", [(0, 1), (True, 1), (2, -1), (2, False)])
def test_invalid_limits_are_rejected(steps, budget):
	with pytest.raises(ValueError, match="refresh"):
		DeltaState(None, max_steps=steps, anchor_bytes=budget)


def test_row_depth_metadata_routes_duplicate_offsets_without_contaminating_cold_rows():
	with _state(row_refresh=True) as (state, _workers):
		catalog = state.service.aggregate_catalog
		key = catalog.selections(1, torch.arange(6), 0)[0].key
		other = catalog.selections(2, torch.arange(6), 0)[0].key
		depths = torch.tensor([0, 1, 2, 0, 2, 1])
		state.record(key, depths)
		depths.fill_(2)
		state.record(other, 1)
		groups = (
			RowSelection(key, torch.tensor([0, 2, 3]), torch.tensor([3, 1, 3])),
			RowSelection(other, torch.tensor([1]), torch.tensor([5])),
		)
		actual = state.row_depths(groups, 4)
		assert actual.tolist() == [0, 1, 1, 0]
		assert state.depth([key, other]) == 2
		actual.fill_(2)
		assert state.row_depths(groups, 4).tolist() == [0, 1, 1, 0]
		assert state.metadata_bytes == 6 * 8


def test_obsolete_row_depth_tensors_are_released_after_their_update_context():
	with _state(row_refresh=True) as (state, _workers):
		service, nodes = state.service, torch.arange(6)
		key = service.aggregate_catalog.selections(1, nodes, 0)[0].key
		assert state.metadata_bytes == 0
		state.record(key, torch.ones(6, dtype=torch.int64))
		state.prepare_features(nodes)
		with state.updating():
			service._begin_version(1, {}, pending_layers=(1, 2))
			state.active.capture(1, nodes, 1)
			service._finish_analysis(1, nodes, 1)
			service.ensure(nodes, 1)
			assert state.metadata_bytes == 48
		assert state.metadata_bytes == 0


@pytest.mark.parametrize(
	"depths",
	[
		torch.tensor([-1, 0, 0, 0, 0, 0]),
		torch.tensor([3, 0, 0, 0, 0, 0]),
		torch.zeros(5, dtype=torch.int64),
		torch.zeros(6),
	],
)
def test_invalid_row_depths_cannot_replace_certified_metadata(depths):
	with _state(row_refresh=True) as (state, _workers):
		key = state.service.aggregate_catalog.selections(1, torch.arange(6), 0)[0].key
		state.record(key, 1)
		with pytest.raises(ValueError, match="Row depths"):
			state.record(key, depths)
		assert state.depth([key]) == 1
