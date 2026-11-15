from lib.profiling.network import NetworkSnapshot, delta


def test_network_delta_reports_interface_and_total_bytes():
	before = NetworkSnapshot(
		"host",
		{
			"ibs0": {
				"bytes_sent": 10,
				"bytes_recv": 20,
				"packets_sent": 1,
				"packets_recv": 2,
				"errin": 0,
				"errout": 0,
				"dropin": 0,
				"dropout": 0,
			}
		},
	)
	after = NetworkSnapshot(
		"host",
		{
			"ibs0": {
				"bytes_sent": 110,
				"bytes_recv": 220,
				"packets_sent": 11,
				"packets_recv": 22,
				"errin": 0,
				"errout": 0,
				"dropin": 0,
				"dropout": 0,
			}
		},
	)
	result = delta(before, after)
	assert result["interfaces"]["ibs0"]["bytes_sent"] == 100
	assert result["totals"]["bytes_recv"] == 200


def test_network_delta_rejects_mixed_hosts():
	value = NetworkSnapshot("a", {})
	try:
		delta(value, NetworkSnapshot("b", {}))
	except ValueError as error:
		assert "different hosts" in str(error)
	else:
		raise AssertionError("mixed-host network snapshots must be rejected")
