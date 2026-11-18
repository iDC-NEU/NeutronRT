import json

from scripts.evaluate.cohort import run_trial


def test_nonzero_trial_retains_actual_exit_status(tmp_path):
	result = run_trial(["{python}", "-c", "raise SystemExit(7)"], tmp_path / "trial", 10, 5)
	assert result["returncode"] == 7
	assert not result["timed_out"]
	assert result["phase"] == "unknown"


def test_startup_deadline_terminates_trial_and_retains_phase(tmp_path):
	result = run_trial(["{python}", "-c", "import time; time.sleep(60)"], tmp_path / "trial", 10, 0.1)
	assert result["timed_out"]
	assert result["timeout_kind"] == "startup"
	assert result["returncode"] != 0


def test_ready_trial_uses_wall_deadline(tmp_path):
	code = (
		"import pathlib,time; pathlib.Path(r'{trial}/phase.json').write_text("
		'\'{"phase":"initialization","method":"RTEC-Full"}\'); time.sleep(60)'
	)
	result = run_trial(["{python}", "-c", code], tmp_path / "trial", 2, 0.5)
	assert result["timeout_kind"] == "wall"
	assert result["phase"] == "initialization"
	assert json.loads((tmp_path / "trial/process-status.json").read_text())["method"] == "RTEC-Full"
