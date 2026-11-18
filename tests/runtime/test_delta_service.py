"""Serving uses certified prior sums and refreshes or falls back when needed."""

from contextlib import ExitStack, contextmanager

import dgl
import pytest
import torch

from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from tests.runtime.test_prepared import _actors
from tests.runtime.test_retained_service import _check_retained


@contextmanager
def _service(*, coalesced=True, gather=False, peer=True, pipeline=False, **kwargs):
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors(coalesced=coalesced, gather=gather))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
		features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
		service = DistributedEmbeddingService(
			graph,
			features,
			(4, 4, 3),
			plane,
			retain_aggregates=True,
			incremental_aggregates=True,
			batch_size=2,
			pipeline=pipeline,
			gather_inputs=gather,
			peer_fetch=peer,
			**kwargs,
		)
		yield service, workers[0].model


def _check(service, model):
	_check_retained(service, model, service.graph, service.features)
	assert not service.remote_plane._retained_states
	assert not service.remote_plane._pending_retire
	assert service._delta.state.active is None


@pytest.mark.parametrize(
	"coalesced,gather,peer", [(False, False, True), (True, False, True), (True, True, True), (True, True, False)]
)
@pytest.mark.parametrize("pipeline", [False, True])
def test_warm_updates_avoid_all_full_neighbor_blocks(coalesced, gather, peer, pipeline, monkeypatch):
	with _service(coalesced=coalesced, gather=gather, peer=peer, pipeline=pipeline) as (service, model):
		_check(service, model)
		monkeypatch.setattr(service.affected, "block", lambda *_a, **_kw: pytest.fail("Warm delta built a block"))
		service.change_features(torch.tensor([0, 3]), torch.full((2, 4), 0.5))
		_check(service, model)
		service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		_check(service, model)
		service.delete_edges(torch.tensor([0, 0, 4]), torch.tensor([1, 1, 5]))
		_check(service, model)
		before = service.counters.tasks
		service.change_features(torch.empty(0, dtype=torch.int64), torch.empty(0, 4))
		assert service.counters.tasks == before
		assert service.counters.delta_tasks > 0 and service.counters.delta_routed_edges > 0
		assert service.counters.delta_fallback_tasks == 0
		assert service.catalog.version == service.aggregate_catalog.version == 4


@pytest.mark.parametrize("reason", ["cold", "budget", "old-h", "old-sum"])
def test_missing_history_and_small_budget_fall_back_to_full_recomputation(reason):
	with _service(delta_anchor_bytes=0 if reason == "budget" else 4096) as (service, model):
		if reason != "cold":
			_check(service, model)
		if reason in ("old-h", "old-sum"):
			catalog = service.catalog if reason == "old-h" else service.aggregate_catalog
			key = catalog.selections(1, torch.tensor([0]), 0)[0].key
			service.remote_plane.retire(key)
		service.change_features(torch.tensor([0]), torch.ones(1, 4))
		_check(service, model)
		assert service.counters.delta_fallback_tasks > 0
		labels = service.profile()["fallback_reasons"]["layers"]
		expected = {
			"cold": "missing_aggregate_anchor",
			"budget": "cold_or_unavailable_state",
			"old-h": "missing_previous_embedding",
			"old-sum": "missing_aggregate_anchor",
		}[reason]
		assert any(expected in values for values in labels.values())
		if reason == "old-h":
			assert service.counters.delta_tasks > 0


def test_refresh_interval_bounds_correction_chains_over_repeated_updates():
	with _service(delta_max_steps=2, pipeline=True) as (service, model):
		_check(service, model)
		for index in range(24):
			service.change_features(torch.tensor([0, 3]), torch.full((2, 4), (index % 5) / 5))
			_check(service, model)
		assert service.counters.delta_refresh_rows > 0 and service.counters.delta_tasks > 0
		assert max(service._delta.state._depths.values()) <= 2


@pytest.mark.parametrize("coalesced,gather", [(False, False), (True, True)])
def test_sparse_admission_reserves_receive_and_compute_workspace_before_dispatch(monkeypatch, coalesced, gather):
	from lib.runtime.aggregates.delta import delta_workspace

	with _service(coalesced=coalesced, gather=gather) as (service, model):
		_check(service, model)
		service.batch_size = 6
		for worker in service.remote_plane.remotes.values():
			worker.max_bytes = 1200
			# Exercise CUDA admission metadata with the CPU-only FIFO actor harness.
			worker.device = torch.device("cuda:0")
		execution = service._delta
		submit = execution._submit
		admissions = []

		def checked_submit(layer, plan, target, tables, next_depth):
			node = service.remote_plane.remotes[target]
			workspace = delta_workspace(plan, service.widths, layer, tables)
			assert node.bytes_used + workspace.required_bytes <= node.max_bytes
			admissions.append(workspace.required_bytes)
			return submit(layer, plan, target, tables, next_depth)

		monkeypatch.setattr(execution, "_submit", checked_submit)
		service.change_features(torch.tensor([0, 3]), torch.ones(2, 4))
		_check(service, model)
		service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		_check(service, model)
		service.delete_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
		_check(service, model)
		assert admissions and service.counters.delta_workspace_splits > 0
		assert not service.counters.delta_workspace_fallbacks


def test_failed_publication_releases_history_and_resume_does_not_reinsert_edges(monkeypatch):
	with _service() as (service, model):
		_check(service, model)
		original = service.aggregate_catalog.publish
		failed = False

		def fail_once(key):
			nonlocal failed
			if not failed:
				failed = True
				raise ValueError("sum publication")
			return original(key)

		monkeypatch.setattr(service.aggregate_catalog, "publish", fail_once)
		before = service.graph.num_edges()
		with pytest.raises(ValueError, match="sum publication"):
			service.insert_edges(torch.tensor([0, 0]), torch.tensor([1, 1]))
		assert service._delta.state.active is None and not service.remote_plane._retained_states
		service.resume_update(1)
		_check(service, model)
		assert service.graph.num_edges() == before + 2
		assert service.catalog.version == service.counters.updates == 1


@pytest.mark.parametrize("kind", ["features", "insert", "delete"])
def test_update_duration_includes_pre_mutation_history_capture(kind, monkeypatch):
	with _service() as (service, model):
		_check(service, model)
		clock = [0.0]
		monkeypatch.setattr("lib.graph.serving.perf_counter", lambda: clock[0])
		monkeypatch.setattr("lib.runtime.service.perf_counter", lambda: clock[0])
		state = service._delta.state
		name = "prepare_features" if kind == "features" else "prepare_edges"
		capture, complete = getattr(state, name), service._complete_update

		def snapshot(*args, **kwargs):
			capture(*args, **kwargs)
			clock[0] = 10.0

		def finish(*args, **kwargs):
			clock[0] = 20.0
			return complete(*args, **kwargs)

		monkeypatch.setattr(state, name, snapshot)
		monkeypatch.setattr(service, "_complete_update", finish)
		if kind == "features":
			result = service.change_features(torch.tensor([0]), torch.ones(1, 4))
		else:
			method = service.insert_edges if kind == "insert" else service.delete_edges
			result = method(torch.tensor([0]), torch.tensor([1]))
		assert result.seconds == service.counters.update_seconds == 20.0
		_check(service, model)


def test_row_refresh_recomputes_only_expired_rows_and_preserves_fresh_deltas(monkeypatch):
	with _service(delta_max_steps=2, row_refresh=True, pipeline=True) as (service, model):
		_check(service, model)
		service.batch_size = 6
		state = service._delta.state
		for group in service.aggregate_catalog.selections(1, torch.tensor([0, 2]), 0):
			depths = torch.zeros(service.remote_plane._shapes[group.key][0], dtype=torch.int64)
			depths[group.rows] = 2
			state.record(group.key, depths)
		blocks = []
		original = service._run_task

		def run_task(task, version):
			blocks.append((task.layer, task.targets.tolist()))
			return original(task, version)

		monkeypatch.setattr(service, "_run_task", run_task)
		service.change_features(torch.tensor([0, 2]), torch.ones(2, 4))
		_check(service, model)
		assert blocks == [(1, [0, 2])]
		assert service.counters.delta_partial_refreshes == 1 and service.counters.delta_refresh_rows == 2
		groups = service.aggregate_catalog.selections(1, torch.tensor([0, 1, 2]), 1)
		assert state.row_depths(groups, 3).tolist() == [0, 1, 0]
		assert service.counters.delta_tasks > 0


def test_row_refresh_stays_within_each_row_limit_over_repeated_feature_and_edge_updates():
	with _service(delta_max_steps=2, row_refresh=True, pipeline=True) as (service, model):
		_check(service, model)
		for index in range(24):
			if index % 3 == 0:
				service.change_features(torch.tensor([0, 3]), torch.full((2, 4), (index % 5) / 5))
			elif index % 3 == 1:
				service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
			else:
				service.delete_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
			_check(service, model)
			assert service._delta.state.depth(list(service._delta.state._depths)) <= 2
		assert service.counters.delta_partial_refreshes > 0
