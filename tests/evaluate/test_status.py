import json

from scripts.evaluate.status import summarize


def test_status_preserves_incomplete_trials_and_edge_network_metrics(tmp_path):
	result = tmp_path / "run"
	result.mkdir()
	(result / "summary.json").write_text(json.dumps({"trials": [{"file": "sage-NrtInc-0.json", "method": "NrtInc"}]}))
	(result / "sage-NrtInc-0.json").write_text(
		json.dumps(
			{
				"valid": False,
				"trace_digest": "abc",
				"workload_counts": {
					"measured": {"input_changes": 10},
					"effective_edge_changes": {"total_effective_edge_delta": 2, "total_effective_edge_changes": 4},
				},
				"profile": {
					"publication": {"service_drained": False},
					"network": {"totals": {"bytes_sent": 30, "bytes_recv": 40}},
				},
			}
		)
	)
	report = summarize(result)
	assert report["state"] == "incomplete"
	assert report["trials"][0]["effective_edge_delta"] == 2
	assert report["trials"][0]["network_bytes_recv"] == 40
