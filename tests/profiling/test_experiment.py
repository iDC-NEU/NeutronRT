import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from lib.profiling.experiment import (
	ProfileCase,
	latency_summary,
	make_trace,
	paired_comparison,
	read_cases,
	tensor_byte_chunks,
	trace_digest,
)


@pytest.mark.parametrize("layout", ["contiguous", "column-major", "strided", "expanded", "empty", "scalar"])
def test_fingerprints_match_logical_values_without_changing_storage(layout):
	value = torch.arange(60, dtype=torch.float32).reshape(5, 12)
	value = {
		"contiguous": value,
		"column-major": value.T,
		"strided": value[:, ::3],
		"expanded": value[:1, :1].expand(20, 30),
		"empty": value[:0],
		"scalar": value[0, 0],
	}[layout]
	before, pointer = value.clone(), value.data_ptr()
	digest = hashlib.sha256()
	for chunk in tensor_byte_chunks(value, scratch_bytes=28):
		if not value.is_contiguous():
			assert len(chunk) <= 28
		digest.update(chunk)
	assert digest.digest() == hashlib.sha256(value.numpy().tobytes(order="C")).digest()
	assert value.data_ptr() == pointer
	assert torch.equal(value, before)


def test_graph_fingerprint_accepts_arxiv_feature_layout_and_detects_changes():
	import dgl

	from lib.models.spec import ModelSpec
	from scripts.evaluate.profile import _identity

	graph = dgl.graph(([0, 2], [1, 1]), num_nodes=4)
	features = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
	spec = ModelSpec("sage", 3, 4, 2)
	identity = _identity(graph, features, spec)
	assert identity == _identity(graph, features.contiguous(), spec)
	features[2, 1] += 1
	assert identity != _identity(graph, features, spec)


def test_workload_is_reproducible_immutable_and_bounded_to_hot_nodes():
	features = torch.zeros(100, 8)
	options = {"updates": 9, "changes": 3, "pool_size": 12, "mixed": True, "seed": 12}
	trace = make_trace(features, **options)
	assert trace_digest(trace) == trace_digest(make_trace(features, **options))
	assert trace_digest(trace) != trace_digest(make_trace(features, **{**options, "seed": 13}))
	assert [update.kind for update in trace] == ["features", "insert", "delete"] * 3
	assert torch.cat([update.first for update in trace]).unique().numel() <= 12
	for index in (1, 4, 7):
		assert trace[index + 1].first is trace[index].first
		assert trace[index + 1].second is trace[index].second
	before = trace_digest(trace)
	features.add_(3)
	assert trace_digest(trace) == before
	with pytest.raises(ValueError, match="memory budget"):
		make_trace(features, **options, max_bytes=64)


@pytest.mark.parametrize(
	"values",
	[
		[{"name": "same"}, {"name": "same"}],
		[{"name": "a", "reference": "b"}],
		[{"name": "a", "devices": "cpu"}],
		[{"name": "../a"}],
		[{"name": "a", "batch_size": True}],
		[{"name": "a", "switches": {"pipeline": "false"}}],
		[{"name": "a", "devices": ["cuda"]}],
		[{"name": "a", "devices": ["cuda:0", "cuda:0"]}],
		[{"name": "a", "unknown": 1}],
	],
)
def test_cases_reject_ambiguous_or_invalid_configuration_before_startup(values):
	with pytest.raises(ValueError):
		read_cases(values)


def test_measured_settings_only_enable_observer_and_allow_cpu_worker_topologies():
	case = read_cases([{"name": "serial", "devices": ["cpu", "cpu"]}])[0]
	assert case.switches.enabled == ()
	assert case.measured_switches.enabled == ("profiling",)
	assert replace(case, switches=case.measured_switches).name == "serial"
	assert ProfileCase("ok").batch_size == 8192
	assert read_cases([{"name": "pipeline", "switches": {"pipeline": True}}])[0].switches.enabled == ("pipeline",)


def _runs(seconds):
	return [
		{
			"repetition": index,
			"valid": True,
			"trace_digest": "same",
			"identity": "dataset/model",
			"client_seconds": seconds,
			"trace_pass": False,
		}
		for index in range(5)
	]


def test_paired_verdict_records_regressions_and_requires_valid_matching_untraced_runs():
	assert paired_comparison(_runs(2), _runs(1))["verdict"] == "gain"
	assert paired_comparison(_runs(1), _runs(2))["verdict"] == "regression"
	assert paired_comparison(_runs(1), _runs(1.01))["verdict"] == "inconclusive"
	assert paired_comparison(_runs(2)[:1], _runs(1)[:1])["verdict"] == "inconclusive"
	for changes in (
		{"valid": False},
		{"trace_pass": True},
		{"trace_digest": "different"},
		{"identity": "different"},
		{"client_seconds": float("nan")},
		{"repetition": 6},
	):
		candidate = _runs(1)
		candidate[0].update(changes)
		with pytest.raises(ValueError):
			paired_comparison(_runs(2), candidate)


def test_latency_percentiles_keep_small_samples_and_empty_query_stream_explicit():
	assert latency_summary([])["p99_seconds"] is None
	assert latency_summary([4, 1, 2, 3]) == {"count": 4, "p50_seconds": 2, "p95_seconds": 4, "p99_seconds": 4}


def test_storage_case_count_is_independent_of_compute_workers_and_requires_its_switch():
	case = ProfileCase.from_dict(
		{
			"name": "storage",
			"devices": ["cpu", "cpu"],
			"storage_partitions": 3,
			"switches": {"distributed_storage": True},
		}
	)
	assert case.storage_partitions == 3 and len(case.devices) == 2
	for value in (0, True, 1.5, -1):
		with pytest.raises(ValueError):
			ProfileCase.from_dict({"name": "bad", "storage_partitions": value})
	with pytest.raises(ValueError, match="distributed_storage"):
		ProfileCase(name="bad", storage_partitions=3)


@pytest.mark.parametrize(
	"options",
	[{"delta_max_steps": 0}, {"delta_max_steps": True}, {"delta_anchor_bytes": -1}, {"delta_anchor_bytes": False}],
)
def test_sparse_refresh_and_history_budgets_are_checked_before_profiling(options):
	with pytest.raises(ValueError, match="case delta_"):
		ProfileCase.from_dict({"name": "bad", **options})


def test_sparse_settings_are_retained_as_trial_identity():
	case = ProfileCase.from_dict({"name": "refresh", "delta_max_steps": 8, "delta_anchor_bytes": 0})
	assert case.delta_max_steps == 8 and case.delta_anchor_bytes == 0


def test_checked_in_sparse_comparisons_keep_equal_topology_and_explicit_refresh_intervals():
	root = Path(__file__).resolve().parents[2]
	for name, steps in (
		("delta-cpu.json", [8, 32]),
		("delta-accuracy-cpu.json", [2, 32]),
	):
		cases = read_cases(json.loads((root / "examples" / "profiles" / name).read_text()))
		assert {case.storage_partitions for case in cases} == {3}
		assert {case.devices for case in cases} == {("cpu", "cpu")}
		assert [case.delta_max_steps for case in cases if case.switches.incremental_aggregates] == steps
		assert all(case.switches.retain_aggregates for case in cases if case.switches.incremental_aggregates)


def test_row_refresh_comparisons_isolate_the_new_policy_under_matching_budgets():
	root = Path(__file__).resolve().parents[2]
	for name in ("row-refresh-cpu.json", "row-refresh-accuracy-cpu.json"):
		cases = read_cases(json.loads((root / "examples" / "profiles" / name).read_text()))
		assert [case.name for case in cases] == ["embedding-only", "chunk-refresh-32", "row-refresh-32"]
		assert cases[1].delta_max_steps == cases[2].delta_max_steps == 32
		assert cases[1].delta_anchor_bytes == cases[2].delta_anchor_bytes
		assert cases[2].reference == cases[1].name
		assert cases[2].switches.as_dict() == {**cases[1].switches.as_dict(), "row_refresh": True}


def test_aggregate_delta_comparisons_isolate_retention_and_keep_embedding_and_full_recipe_references():
	root = Path(__file__).resolve().parents[2]
	for name in ("aggregate-delta-cpu.json", "aggregate-delta-accuracy-cpu.json"):
		cases = read_cases(json.loads((root / "examples" / "profiles" / name).read_text()))
		embedding, retained, full, sparse = cases
		assert not embedding.switches.retain_aggregates
		assert full.switches.aggregate_only and not full.switches.incremental_aggregates
		assert sparse.reference == retained.name
		assert sparse.switches.as_dict() == {**retained.switches.as_dict(), "aggregate_only": True}
		assert sparse.delta_max_steps == retained.delta_max_steps == 32
		assert sparse.delta_anchor_bytes == retained.delta_anchor_bytes
		assert {case.storage_partitions for case in cases} == {3}
		assert {case.devices for case in cases} == {("cpu", "cpu")}
