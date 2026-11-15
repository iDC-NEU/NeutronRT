import pytest

from lib.profiling.metrics import latency_envelope, measurements


def trial(size, seconds, count=20, valid=True):
	return {"valid": valid, "updates": [{"kind": 2, "changes": size, "client_seconds": seconds}] * count}


def test_latency_envelope_requires_every_batch_and_repetition_to_meet_bound():
	rows = [trial(4, 0.5), trial(4, 1.5), trial(2, 0.9), trial(8, 0.1, count=19)]
	result = latency_envelope(rows, bounds=[1, 5])
	assert result[0]["largest_tested_batch"] == 2
	assert result[0]["minimum_edge_updates_per_second"] == 2 / 0.9
	assert result[1]["largest_tested_batch"] == 4
	assert result[1]["minimum_edge_updates_per_second"] == 4 / 1.5


def test_invalid_repetition_disqualifies_whole_batch_size():
	assert latency_envelope([trial(4, 0.1), trial(4, 0.1, valid=False)])[0]["largest_tested_batch"] is None


def test_sparse_reports_include_delta_work_and_index_maintenance():
	worker = {
		"counters": {"model.layer1.compiled_edges": 3, "delta.layer1.routed_edges": 7},
		"timings": {
			"worker.model_cpu.layer1": {"total_seconds": 1},
			"worker.delta_cpu.layer1": {"total_seconds": 2},
			"worker.reconstruction_cpu.layer1": {"total_seconds": 0.5},
			"worker.delta_cuda.layer1": {"total_seconds": 0.3},
		},
		"rss": {"peak_rss_bytes": 64},
		"resident_current_bytes": 16,
	}
	recorded = {
		"method": "NrtInc",
		"neighborhood_policy": {"fanout": -1},
		"updates": [{"changes": 4, "kind": 2, "client_seconds": 0.1}],
		"correctness": [{"mean_squared_error": 0, "max_error": 0, "prediction_disagreements": 0}],
		"profile": {
			"workers": [worker],
			"coordinator": {"rss": {"peak_rss_bytes": 128}, "timings": {"root.graph_formats": {"total_seconds": 4}}},
			"p2p": {"send_payload_bytes": 10},
			"network": {"totals": {"bytes_sent": 100, "bytes_recv": 120}},
			"storage": {"partitions": {"0": {"shared_response_bytes": 32}}},
		},
	}
	result = measurements(recorded)
	assert result["accesses"]["executed_edge_messages"] == 10
	assert result["accesses"]["delta_routed_edges"] == 7
	assert result["phases_seconds"]["worker_model_host"] == 3.5
	assert result["phases_seconds"]["worker_model_cuda"] == pytest.approx(0.3)
	assert result["phases_seconds"]["graph_index_maintenance"] == 4
	assert result["communication"]["storage_selected_adjacency_bytes"] == 32
	assert result["communication"]["network"]["totals"]["bytes_sent"] == 100
