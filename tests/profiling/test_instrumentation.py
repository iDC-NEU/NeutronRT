from lib.profiling.instrumentation import counter_delta, counter_snapshot, grouped_counter_delta, profile_window


def test_nested_counter_diffs_are_copy_safe_and_report_resets():
	before = {"policy": {"waves": 2}, "cache_hits": 4, "unchanged": 7, "flag": True}
	after = {"policy": {"waves": 5}, "cache_hits": 1, "unchanged": 7, "new": 3, "flag": False}
	assert counter_snapshot(before)["policy"] is not before["policy"]
	delta, resets = counter_delta(before, after)
	assert delta == {"policy": {"waves": 3}, "cache_hits": 0, "unchanged": 0, "new": 3}
	assert resets == ["cache_hits"]


def test_counter_groups_only_report_exposed_runtime_names():
	groups = grouped_counter_delta(
		{
			"policy.dispatches": 2,
			"fallback.refreshes": 3,
			"cache.reuse": 4,
			"unclassified": 5,
		}
	)
	assert groups == {
		"policy": {"policy.dispatches": 2},
		"fallback": {"fallback.refreshes": 3},
		"cache": {"cache.reuse": 4},
	}


def test_profile_window_preserves_configuration_and_gauges_without_differencing_them():
	window = profile_window(
		{
			"dispatch": {"mode": "round_robin", "choices": 1, "outstanding_cost_units": {"0": 2}},
			"cache": {"budget_bytes": 100, "data_bytes": 40, "hits": 2},
		},
		{
			"dispatch": {"mode": "locality_load", "choices": 4, "outstanding_cost_units": {"0": 3}},
			"cache": {"budget_bytes": 100, "data_bytes": 60, "hits": 5},
		},
	)
	assert window["configuration"]["dispatch.mode"] == "locality_load"
	assert window["configuration"]["cache.budget_bytes"] == 100
	assert window["gauges"]["cache.data_bytes"] == 60
	assert window["counter_deltas"]["dispatch"]["choices"] == 3
	assert window["counter_deltas"]["cache"]["hits"] == 3


def test_profile_window_counts_lazy_fallback_paths_from_zero_and_keeps_invalidations_as_counters():
	window = profile_window(
		{"fallback_reasons": {}, "cache": {"invalidated_bytes": 12, "data_bytes": 40}},
		{
			"fallback_reasons": {"missing_previous_embedding": 3},
			"cache": {"invalidated_bytes": 20, "data_bytes": 60},
		},
	)
	assert window["counter_deltas"]["fallback_reasons"]["missing_previous_embedding"] == 3
	assert "fallback_reasons.missing_previous_embedding" in window["new_counter_paths"]
	assert window["counter_deltas"]["cache"]["invalidated_bytes"] == 8
	assert window["gauges"]["cache.data_bytes"] == 60

	reset = profile_window({"fallback_reasons": {"old": 9}}, {"fallback_reasons": {"old": 2}})
	assert reset["counter_deltas"]["fallback_reasons"]["old"] == 0
	assert reset["counter_resets"] == ["fallback_reasons.old"]


def test_counter_delta_handles_new_nested_paths_from_an_empty_profile():
	delta, resets = counter_delta({}, {"fallback_reasons": {"cold_or_unavailable_state": 2}})
	assert delta == {"fallback_reasons": {"cold_or_unavailable_state": 2}}
	assert resets == []
