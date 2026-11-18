import json
from types import SimpleNamespace

import pytest

from scripts.evaluate.continue_manifest import step


def test_continuation_waits_then_submits_once_and_reuses_saved_task(tmp_path, monkeypatch):
	manifest = tmp_path / "manifest.json"
	artifact = tmp_path / "data.npz"
	artifact.write_bytes(b"fixture")
	manifest.write_text(json.dumps({"required_files": [str(artifact)]}))
	plan = {"after_tasks": ["old"], "manifest": str(manifest), "output": str(tmp_path / "output")}
	state = {"plan": plan}
	path = tmp_path / "state.json"
	status = {"old": "running", "new": "running"}

	def poll(task):
		return {"task_id": task, "status": status[task], "result": {"returncode": 0}}

	monkeypatch.setattr("scripts.evaluate.continue_manifest.poll", poll)
	calls = []

	def run(*args, **kwargs):
		calls.append(args)
		return SimpleNamespace(stdout='{"task_id":"new"}')

	monkeypatch.setattr("scripts.evaluate.continue_manifest.subprocess.run", run)
	assert not step(plan, state, path)
	assert not calls
	status["old"] = "failed"  # Old wrapper failed, but completed artifact exists.
	assert not step(plan, state, path)
	assert state["task_id"] == "new" and len(calls) == 1
	state = json.loads(path.read_text())
	status["new"] = "completed"
	assert step(plan, state, path)
	assert len(calls) == 1


def test_missing_completed_download_cannot_trigger_new_transfer(tmp_path, monkeypatch):
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps({"required_files": [str(tmp_path / "missing")]}))
	plan = {"after_tasks": ["old"], "manifest": str(manifest), "output": str(tmp_path / "output")}
	monkeypatch.setattr("scripts.evaluate.continue_manifest.poll", lambda task: {"status": "failed"})
	with pytest.raises(FileNotFoundError, match="no job submitted"):
		step(plan, {"plan": plan}, tmp_path / "state.json")


def test_continuation_rejects_configuration_drift_while_waiting(tmp_path, monkeypatch):
	manifest = tmp_path / "manifest.json"
	manifest.write_text('{"required_files": []}')
	plan = {"after_tasks": ["old"], "manifest": str(manifest), "output": str(tmp_path / "output")}
	state = {"plan": plan}
	monkeypatch.setattr("scripts.evaluate.continue_manifest.poll", lambda task: {"status": "running"})
	assert not step(plan, state, tmp_path / "state.json")
	manifest.write_text('{"required_files": [], "changed": true}')
	with pytest.raises(ValueError, match="changed while waiting"):
		step(plan, state, tmp_path / "state.json")
