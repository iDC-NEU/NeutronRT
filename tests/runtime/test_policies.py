from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.policies import DeferredPolicyConfig, DispatchPolicy, FallbackCounters, FeatureRowCache
from tests.graph.test_serving import _setup


@dataclass(frozen=True)
class _Worker:
	owner: Owner


def test_round_robin_is_deterministic_and_locality_load_accounts_for_outstanding_cost():
	workers = [_Worker(Owner(1, 1)), _Worker(Owner(2, 1))]
	round_robin = DispatchPolicy()
	first = round_robin.choose(workers).worker
	second = round_robin.choose(workers).worker
	assert (first.owner.rank, second.owner.rank) == (1, 2)
	round_robin.release(first.owner, 1)
	round_robin.release(second.owner, 1)

	policy = DispatchPolicy("locality_load")
	first = policy.choose(workers, locality_rows={}, total_rows=2, task_cost=1)
	second = policy.choose(workers, locality_rows={}, total_rows=2, task_cost=1)
	assert first.worker.owner.rank == 1
	assert second.worker.owner.rank == 2
	assert policy.profile()["outstanding_cost_units"] == {"1:g1": 1, "2:g1": 1}
	policy.release(first.worker.owner, first.reserved_cost)
	policy.release(second.worker.owner, second.reserved_cost)
	serial = DispatchPolicy("locality_load")
	choice = serial.choose(workers[:1], locality_rows={workers[0].owner: 1}, total_rows=1, task_cost=2)
	assert choice.worker.owner == workers[0].owner
	assert serial.profile()["serial_fallbacks"] == 1
	serial.release(choice.worker.owner, choice.reserved_cost)
	strong = DispatchPolicy("locality_load", locality_weight=10, load_weight=1)
	choice = strong.choose(workers, locality_rows={workers[0].owner: 10}, total_rows=10, task_cost=1)
	assert choice.worker.owner == workers[0].owner
	remaining = [worker for worker in workers if worker != choice.worker]
	other = strong.choose(remaining, locality_rows={workers[0].owner: 10}, total_rows=10, task_cost=9)
	assert other.worker.owner == workers[1].owner
	strong.release(choice.worker.owner, choice.reserved_cost)
	strong.release(other.worker.owner, other.reserved_cost)
	with pytest.raises(RuntimeError, match="underflow"):
		strong.adjust(workers[0].owner, 2, 1)


def test_default_round_robin_does_not_scan_catalog_locality_metadata():
	service, plane, _, _ = _setup("sage")
	try:
		service._locality_rows = Mock(side_effect=AssertionError("round robin locality scan"))
		worker = service._pick(required=torch.tensor([0]), input_layer=1, task_cost=1)
		service._release_dispatch(worker)
		assert service.dispatch_policy.profile()["locality_choices"] == 0
	finally:
		plane.close()


def test_feature_row_cache_enforces_byte_budget_lru_and_exact_version_invalidation():
	cache = FeatureRowCache(64)
	values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
	nodes = torch.tensor([0, 1], dtype=torch.int64)
	first = cache.resolve(0, nodes, values[:2])
	values[0].add_(100)
	torch.testing.assert_close(first, torch.arange(4, dtype=torch.float32).reshape(2, 2))
	second = cache.resolve(0, torch.tensor([1, 0]), values.index_select(0, torch.tensor([1, 0])))
	torch.testing.assert_close(second, torch.tensor([[2.0, 3.0], [0.0, 1.0]]))
	cache.resolve(0, torch.tensor([2]), values[2:])
	profile = cache.profile()
	assert profile["hits"] == 2 and profile["misses"] == 3 and profile["evictions"] == 1
	assert profile["bytes_used"] <= profile["budget_bytes"]
	cache.resolve(1, torch.tensor([0]), values[:1] + 50)
	profile = cache.profile()
	assert profile["invalidations"] == 2
	assert profile["misses"] == 4
	assert profile["bytes_used"] == 32
	assert profile["accounting_scope"] == "tensor_payload_plus_logical_metadata"
	assert not profile["physical_overhead_included"]
	hot = FeatureRowCache(64)
	hot.resolve(0, torch.tensor([0, 1]), values[:2])
	for _ in range(50):
		hot.resolve(0, torch.tensor([0, 1]), values[:2])
	assert hot.profile()["bytes_used"] == 64 and hot.profile()["data_bytes"] == 16
	hot.resolve(0, torch.tensor([2]), values[2:])
	assert hot.profile()["bytes_used"] == 64 and hot.profile()["evictions"] == 1


def test_feature_row_cache_reports_oversized_rows_without_serving_stale_data():
	cache = FeatureRowCache(24)
	values = torch.ones(1, 2)
	result = cache.resolve(0, torch.tensor([4]), values)
	torch.testing.assert_close(result, values)
	assert cache.profile()["oversized_rows"] == 1
	assert cache.profile()["bytes_used"] == 0
	with pytest.raises(ValueError, match="CPU or CUDA float32"):
		cache.resolve(0, torch.tensor([4]), values.double())
	with pytest.raises(ValueError, match="CPU or CUDA float32"):
		cache.resolve(0, torch.tensor([4]), values.reshape(-1))
	with pytest.raises(ValueError, match="int64"):
		cache.resolve(0, torch.tensor([4], dtype=torch.int32), values)


@pytest.mark.parametrize(
	"kwargs",
	[
		{"mode": "version_window", "window_seconds": 1},
		{"mode": "fixed_time"},
		{"mode": "hotspot", "window_seconds": 1, "hotspot_degree_threshold": 1},
		{"mode": "hotspot", "window_seconds": 1, "hotspot_degree_threshold": 1, "hotspot_max_changes": 0},
	],
)
def test_deferred_policy_rejects_unsupported_or_incomplete_options(kwargs):
	with pytest.raises(ValueError):
		DeferredPolicyConfig(**kwargs)


def test_fallback_counters_keep_layer_and_reason_scopes_separate():
	counters = FallbackCounters()
	counters.add(1, "cold_or_unavailable_state", 4)
	counters.add(1, "cold_or_unavailable_state", 2)
	counters.add(2, "missing_previous_embedding", 3)
	assert counters.as_dict() == {
		"scope": "one increment per full-fallback task or selected fallback row",
		"layers": {
			"layer1": {"cold_or_unavailable_state": {"tasks": 2, "rows": 6}},
			"layer2": {"missing_previous_embedding": {"tasks": 1, "rows": 3}},
		},
	}


def test_managed_runtime_rejects_unmatched_cache_and_deferred_switches_before_launch():
	model = ModelSpec("sage", 2, 2, 2)
	features = torch.zeros(1, 2)
	with pytest.raises(ValueError, match="positive byte budget"):
		ManagedRuntime(
			None,
			features,
			model,
			("cpu",),
			switches=RuntimeFeatures(worker_feature_cache=True),
		)
	with pytest.raises(ValueError, match="requires worker_feature_cache"):
		ManagedRuntime(None, features, model, ("cpu",), worker_feature_cache_bytes=64)
	with pytest.raises(NotImplementedError, match="wire protocol"):
		ManagedRuntime(
			None,
			features,
			model,
			("cpu",),
			switches=RuntimeFeatures(worker_feature_cache=True),
			worker_feature_cache_bytes=64,
		)
	with pytest.raises(ValueError, match="bounded_staleness"):
		ManagedRuntime(None, features, model, ("cpu",), deferred_mode="fixed_time", deferred_window_seconds=1)
	with pytest.raises(ValueError, match="Unsupported dispatch policy"):
		ManagedRuntime(None, features, model, ("cpu",), dispatch_mode="made_up")
