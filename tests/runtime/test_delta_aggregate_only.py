"""Sparse serving reconstructs old h while persisting only aggregate state."""

import pytest
import torch

from tests.runtime.test_delta_service import _check, _service


def _drop_h(service):
	for key in tuple(service.remote_plane._shapes):
		if key.kind == "h":
			service.remote_plane.retire(key)


@pytest.mark.parametrize("coalesced,gather,peer", [(False, False, True), (True, True, True), (True, True, False)])
@pytest.mark.parametrize("row_refresh", [False, True])
def test_sparse_updates_reconstruct_old_inputs_even_after_discarding_all_h(coalesced, gather, peer, row_refresh):
	with _service(
		aggregate_only=True,
		coalesced=coalesced,
		gather=gather,
		peer=peer,
		row_refresh=row_refresh,
		delta_max_steps=2,
		pipeline=True,
	) as (service, model):
		_check(service, model)
		for index in range(12):
			_drop_h(service)
			if index % 3 == 0:
				service.change_features(torch.tensor([0, 3]), torch.full((2, 4), index / 10))
			elif index % 3 == 1:
				service.insert_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
			else:
				service.delete_edges(torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5]))
			_check(service, model)
			assert not any(key.kind == "h" for key in service.remote_plane._retained_states)
			for key in service.remote_plane._shapes:
				if key.kind == "h":
					assert service.catalog.current_nodes(key).numel()
		assert service.counters.delta_tasks > 0 and service.counters.reconstructed_rows > 0
		assert service.counters.delta_refresh_rows > 0
		assert not service.counters.delta_workspace_fallbacks


def test_small_workspace_falls_back_without_leaking_temporary_state():
	with _service(aggregate_only=True) as (service, model):
		_check(service, model)
		_drop_h(service)
		for node in service.remote_plane.remotes.values():
			node.max_bytes = 64
		service.change_features(torch.tensor([0, 3]), torch.ones(2, 4))
		_check(service, model)
		assert service.counters.delta_workspace_fallbacks > 0
		assert any(
			"workspace_capacity" in values for values in service.profile()["fallback_reasons"]["layers"].values()
		)
		assert all(
			key.kind == "a" or service.catalog.current_nodes(key).numel() for key in service.remote_plane._shapes
		)


@pytest.mark.parametrize("reason", ["old-sum", "budget"])
def test_uncertified_aggregate_history_uses_full_recomputation(reason):
	with _service(aggregate_only=True, delta_anchor_bytes=0 if reason == "budget" else 4096) as (service, model):
		_check(service, model)
		if reason == "old-sum":
			key = service.aggregate_catalog.selections(1, torch.tensor([0]), 0)[0].key
			service.remote_plane.retire(key)
		_drop_h(service)
		service.change_features(torch.tensor([0]), torch.ones(1, 4))
		_check(service, model)
		assert service.counters.delta_fallback_tasks > 0


def test_interrupted_sparse_aggregate_only_update_cleans_temporaries_and_resumes_once(monkeypatch):
	with _service(aggregate_only=True) as (service, model):
		_check(service, model)
		_drop_h(service)
		original = service.aggregate_catalog.publish
		failed = False

		def fail_once(key):
			nonlocal failed
			if key.layer == 2 and not failed:
				failed = True
				raise RuntimeError("aggregate publication")
			return original(key)

		monkeypatch.setattr(service.aggregate_catalog, "publish", fail_once)
		before = service.graph.num_edges()
		with pytest.raises(RuntimeError, match="aggregate publication"):
			service.insert_edges(torch.tensor([0, 0]), torch.tensor([1, 1]))
		assert not service.remote_plane._retained_states and service._delta.state.active is None
		assert all(
			key.kind == "a" or service.catalog.current_nodes(key).numel() for key in service.remote_plane._shapes
		)
		service.resume_update(1)
		_check(service, model)
		assert service.graph.num_edges() == before + 2


def test_reconstruction_resolves_prefix_after_admission_discards_cached_h(monkeypatch):
	from contextlib import ExitStack

	with _service(aggregate_only=True) as (service, model):
		_check(service, model)
		plane = service.remote_plane
		original = plane.make_room
		evicted = []

		def make_room(owner, required):
			for key in tuple(plane._shapes):
				if key.kind == "h":
					evicted.append(key)
					plane.retire(key)
			original(owner, required)

		monkeypatch.setattr(plane, "make_room", make_room)
		with ExitStack() as stack:
			service._aggregate_only.reconstruct(torch.tensor([0, 1]), 2, stack)
		assert evicted
		_check(service, model)
