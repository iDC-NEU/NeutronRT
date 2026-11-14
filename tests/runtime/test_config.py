import json
from dataclasses import fields, replace

import pytest

from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime


def test_switches_round_trip_and_can_be_enabled_independently_from_the_exact_baseline():
	baseline = RuntimeFeatures.baseline()
	assert baseline.mask == 0 and not baseline.enabled
	for index, field in enumerate(fields(baseline)):
		setting = replace(baseline, **{field.name: True})
		assert setting.enabled == (field.name,) and setting.mask == 1 << index
		assert RuntimeFeatures.from_dict(json.loads(json.dumps(setting.as_dict()))) == setting


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_false_like_strings_and_numbers_cannot_silently_enable_a_function(value):
	with pytest.raises(ValueError, match="boolean"):
		RuntimeFeatures.from_dict({"pipeline": value})
	with pytest.raises(ValueError, match="boolean"):
		RuntimeFeatures(pipeline=value)


def test_unknown_switches_fail_instead_of_becoming_unmeasured_noops():
	with pytest.raises(ValueError, match="Unknown"):
		RuntimeFeatures.from_dict({"pipline": True})


@pytest.mark.parametrize(
	"options",
	[
		{"max_stale_versions": True},
		{"max_stale_versions": -1},
		{"idle_flush_seconds": float("nan")},
		{"idle_flush_seconds": 0},
	],
)
def test_invalid_staleness_parameters_fail_before_graph_access_or_worker_startup(options):
	with pytest.raises(ValueError, match="staleness"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], **options)


@pytest.mark.parametrize(
	"options",
	[
		{"storage_partitions": 0},
		{"storage_partitions": True},
		{"storage_overlay_bytes": -1},
		{"storage_response_bytes": -1},
	],
)
def test_invalid_cpu_storage_limits_fail_before_graph_access_or_process_startup(options):
	with pytest.raises(ValueError, match="storage"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], **options)


def test_native_and_sharded_topology_cannot_silently_use_the_old_seed_graph():
	with pytest.raises(NotImplementedError, match="sharded"):
		ManagedRuntime(
			None,
			None,
			ModelSpec("sage", 4, 4, 3),
			["cpu"],
			switches=RuntimeFeatures(native_analysis=True, distributed_storage=True),
		)


def test_multiple_storage_partitions_require_the_explicit_switch():
	with pytest.raises(ValueError, match="distributed_storage"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], storage_partitions=2)


@pytest.mark.parametrize(
	"options",
	[{"delta_max_steps": 0}, {"delta_max_steps": True}, {"delta_anchor_bytes": -1}, {"delta_anchor_bytes": False}],
)
def test_invalid_delta_limits_fail_before_graph_access_or_worker_startup(options):
	with pytest.raises(ValueError, match="Delta refresh"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], **options)


def test_sparse_updates_require_aggregate_state_before_startup():
	switches = RuntimeFeatures(incremental_aggregates=True)
	with pytest.raises(ValueError, match="retain_aggregates"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], switches=switches)


def test_row_refresh_requires_sparse_execution_before_graph_access_or_startup():
	with pytest.raises(ValueError, match="Row refresh requires"):
		ManagedRuntime(None, None, ModelSpec("sage", 4, 4, 3), ["cpu"], switches=RuntimeFeatures(row_refresh=True))
