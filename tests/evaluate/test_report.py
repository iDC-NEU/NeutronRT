from scripts.evaluate.report import comparison_table


def test_comparison_table_separates_p2p_and_nic_counters():
	row = {
		"cohort": "products",
		"model": "sage",
		"method": "NrtInc",
		"fanout": -1,
		"valid": True,
		"response_mean_seconds": 2.0,
		"embedding_mse": 0.0,
		"prediction_disagreement_fraction": 0.0,
		"p2p_payload_bytes": 1024**2,
		"network_bytes_sent": 3 * 1024**2,
		"network_bytes_recv": 4 * 1024**2,
	}
	table = comparison_table({"rows": [row]})
	assert "P2P payload (MiB)" in table
	assert "NIC sent (MiB)" in table
	assert "| products | sage | NrtInc | 1.00 | 3.00 | 4.00 |" in table
