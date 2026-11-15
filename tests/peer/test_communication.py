import pytest

from scripts.peer.communication import (
	RESULT_MARKER,
	_parse_result,
	_quantile,
	_read_perftest_report,
	_worker_command,
)


def test_quantile_interpolates_samples():
	assert _quantile([1.0, 2.0, 4.0, 8.0], 0.5) == 3.0
	assert _quantile([1.0, 2.0, 4.0, 8.0], 0.95) == pytest.approx(7.4)


def test_parse_result_uses_the_rank_zero_marker():
	assert _parse_result(f'warning\n{RESULT_MARKER}{{"ok": true}}\n') == {"ok": True}


def test_worker_command_repeats_payload_sizes():
	command = _worker_command(
		"python",
		"gloo-cpu",
		0,
		"10.66.0.4",
		29600,
		"ibs10",
		0,
		(1024, 4096),
		5,
		10,
		30,
	)
	assert command.count("--size") == 2
	assert command[command.index("--mode") + 1] == "gloo-cpu"


def test_perftest_report_accepts_trailing_comma(tmp_path):
	report = tmp_path / "perftest.json"
	report.write_text('{"results": {"bw_avg": 18.7,}}\n')

	parsed, note = _read_perftest_report(report)

	assert parsed["results"]["bw_avg"] == 18.7
	assert note is not None
