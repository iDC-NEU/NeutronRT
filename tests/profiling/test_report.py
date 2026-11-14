from lib.profiling.report import markdown_report


def test_summary_preserves_failed_cases_and_does_not_hide_accuracy_costs():
	valid = {
		"case": "optimized",
		"valid": True,
		"client_changes_per_second": 20,
		"profile": {
			"engine": {"completed_changes_per_second": 30, "rows": 5},
			"workers": [{"counters": {"model.layer1.compiled_edges": 10}}],
			"p2p": {"send_payload_bytes": 1024**2},
		},
		"query_latency": {"p99_seconds": 0.02},
		"query_accuracy": [{"max_abs_error": 0.5, "accuracy_loss_percentage_points": 2.0}],
	}
	report = markdown_report(
		{"runs": [valid, {"case": "failed", "valid": False}], "comparisons": [{"case": "optimized", "verdict": "gain"}]}
	)
	assert "optimized | 1/1 | 20.00 | 30.00 | 5 | 10 | 0 | 0 | 1.00 | 20.00 | 0.5 | 2.000 | gain" in report
	assert "failed | 0/1" in report and "invalid" in report
	assert "rather than the entire official test split" in report


def test_summary_separates_delta_work_and_refresh_from_full_neighbor_edges():
	run = {
		"case": "sparse",
		"valid": True,
		"client_changes_per_second": 20,
		"profile": {
			"engine": {"completed_changes_per_second": 30, "rows": 5, "delta_routed_edges": 4, "delta_refresh_rows": 2},
			"workers": [{"counters": {"model.layer1.compiled_edges": 10}}],
			"p2p": {"send_payload_bytes": 1024**2},
		},
		"query_latency": {"p99_seconds": 0.02},
	}
	report = markdown_report({"runs": [run], "comparisons": []})
	assert "5 | 10 | 4 | 2 | 1.00" in report
