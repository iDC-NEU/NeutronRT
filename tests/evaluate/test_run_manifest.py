import json
from types import SimpleNamespace

import pytest

from scripts.evaluate.run_manifest import main


def test_submission_uses_frozen_resources_and_explicit_workdir(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	manifest = tmp_path / "cohort.json"
	manifest.write_text(json.dumps({"hardware_profile": "cpu-test", "command": ["{python}", "-V"]}))
	hardware = tmp_path / "hardware.json"
	hardware.write_text(json.dumps({"profiles": {"cpu-test": {"cpu": 16, "mem": 32, "gpus": 0}}}))
	calls = []

	def run(command, **kwargs):
		calls.append(command)
		return SimpleNamespace(stdout='{"task_id":"known-id"}', returncode=0, stderr="")

	def freeze(root, destination):
		destination.mkdir(parents=True)
		return {"commit": "fixture", "files": {}}

	monkeypatch.setattr("lib.profiling.submission.freeze", freeze)
	monkeypatch.setattr("scripts.evaluate.run_manifest.subprocess.run", run)
	monkeypatch.setattr(
		"sys.argv", ["run_manifest", str(manifest), "--output", "fresh", "--hardware", str(hardware), "--submit"]
	)
	main()
	assert len(calls) == 1
	command = calls[0]
	assert command[command.index("--cpu") + 1] == "16"
	assert command[command.index("--mem") + 1] == "32"
	assert command[command.index("--device-num") + 1] == "0"
	assert "; cd " in command[-1] and " && exec " in command[-1]
	assert json.loads((tmp_path / "logs/task-ledger/known-id.json").read_text())["profile"]["gpus"] == 0
	# A queued task has not created its output yet. A second submission must
	# nevertheless be rejected and must not contact the scheduler again.
	main()  # Return the recorded task ID; do not submit again.
	assert len(calls) == 1
	frozen_path = next((tmp_path / "logs/submissions").glob("*/manifest.json"))
	manifest.write_text(json.dumps({"hardware_profile": "cpu-test", "command": ["changed"]}))
	assert json.loads(frozen_path.read_text())["hardware_profile"] == "cpu-test"
	(tmp_path / "existing").mkdir()
	monkeypatch.setattr(
		"sys.argv", ["run_manifest", str(manifest), "--output", "existing", "--hardware", str(hardware), "--submit"]
	)
	with pytest.raises(FileExistsError):
		main()
	assert len(calls) == 1


def test_changed_frozen_manifest_is_rejected_before_starting_work(tmp_path, monkeypatch):
	manifest = tmp_path / "manifest.json"
	manifest.write_text('{"command": ["changed"]}')
	monkeypatch.setattr(
		"sys.argv",
		["run_manifest", str(manifest), "--output", str(tmp_path / "output"), "--expected-manifest-sha256", "0" * 64],
	)
	with pytest.raises(ValueError, match="Frozen submission manifest changed"):
		main()
	assert not (tmp_path / "output").exists()


def test_changed_checkpoint_is_rejected_before_starting_work(tmp_path, monkeypatch):
	checkpoint = tmp_path / "weights.pt"
	checkpoint.write_bytes(b"changed")
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps({"expected_inputs_sha256": {str(checkpoint): "0" * 64}}))
	monkeypatch.setattr("sys.argv", ["run_manifest", str(manifest), "--output", str(tmp_path / "output")])
	with pytest.raises(ValueError, match="Frozen input changed"):
		main()
	assert not (tmp_path / "output").exists()


def test_early_exit_137_is_not_reported_as_wall_timeout(tmp_path, monkeypatch):
	import subprocess

	monkeypatch.chdir(tmp_path)
	subprocess.run(["git", "init", "-q"], check=True)
	(tmp_path / "fixture").write_text("fixture")
	subprocess.run(["git", "add", "fixture"], check=True)
	subprocess.run(
		[
			"git",
			"-c",
			"user.name=Test",
			"-c",
			"user.email=test@example.invalid",
			"-c",
			"core.hooksPath=/dev/null",
			"commit",
			"-qm",
			"fixture",
		],
		check=True,
	)
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps({"wall_seconds": 30, "command": ["{python}", "-c", "raise SystemExit(137)"]}))
	monkeypatch.setattr("sys.argv", ["run_manifest", str(manifest), "--output", str(tmp_path / "output")])
	with pytest.raises(SystemExit) as error:
		main()
	assert error.value.code == 137
	status = json.loads((tmp_path / "output/process-status.json").read_text())
	assert not status["timed_out"] and status["phase"] == "unknown"
