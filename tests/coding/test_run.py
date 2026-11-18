import json
import subprocess

import pytest
import typer

from scripts.coding.run import main


class _FakeProcess:
	def __init__(self, lines, returncode=0):
		self.stdout = iter(lines)
		self.returncode = returncode
		self.terminated = False

	def __enter__(self):
		return self

	def __exit__(self, *args):
		return False

	def wait(self, timeout=None):
		return self.returncode

	def poll(self):
		return self.returncode

	def terminate(self):
		self.terminated = True

	def kill(self):
		self.returncode = -9


def test_structured_failure_event_propagates_and_publishes_failed_state(monkeypatch, tmp_path):
	prompt = tmp_path / "prompt.md"
	prompt.write_text("finish the task")
	process = _FakeProcess(
		[
			json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\n",
			json.dumps({"type": "error", "message": "service failed"}) + "\n",
		]
	)
	monkeypatch.setattr("scripts.coding.run.subprocess.Popen", lambda *args, **kwargs: process)

	with pytest.raises(typer.Exit) as error:
		main(prompt, tmp_path / "output", profile="test", cwd=tmp_path)

	assert error.value.exit_code == 1
	state = json.loads((tmp_path / "output/session.json").read_text())
	assert state["state"] == "failed"
	assert state["returncode"] == 1
	assert state["thread_id"] == "thread-1"
	assert state["error"] == "service failed"


def test_process_start_failure_publishes_failed_state(monkeypatch, tmp_path):
	prompt = tmp_path / "prompt.md"
	prompt.write_text("finish the task")

	def start(*args, **kwargs):
		raise OSError("codex missing")

	monkeypatch.setattr("scripts.coding.run.subprocess.Popen", start)
	with pytest.raises(OSError, match="codex missing"):
		main(prompt, tmp_path / "output", profile="test", cwd=tmp_path)

	state = json.loads((tmp_path / "output/session.json").read_text())
	assert state["state"] == "failed"
	assert "codex missing" in state["error"]


def test_terminal_completion_queues_bounded_parent_callback_and_records_receipt(monkeypatch, tmp_path):
	prompt = tmp_path / "prompt.md"
	prompt.write_text("finish the task")
	process = _FakeProcess([json.dumps({"type": "thread.started", "thread_id": "child-thread"}) + "\n"])
	queued = []

	class Result:
		returncode = 0
		stdout = "queued with warning\n"
		stderr = "warning: active writer\n"

	def run(args, **kwargs):
		queued.append((args, kwargs))
		return Result()

	monkeypatch.setattr("scripts.coding.run.subprocess.Popen", lambda *args, **kwargs: process)
	monkeypatch.setattr("scripts.coding.run.subprocess.run", run)
	with pytest.raises(typer.Exit) as error:
		main(prompt, tmp_path / "output", profile="test", cwd=tmp_path, notify_thread="parent-thread")

	assert error.value.exit_code == 0
	receipt = json.loads((tmp_path / "output/callback-receipt.json").read_text())
	assert receipt["status"] == "queued"
	assert receipt["queue_warning"]
	assert not receipt["delivery_verified"]
	assert queued and queued[0][0][:5] == ["codex", "queue", "--resume", "--thread", "parent-thread"]
	assert (tmp_path / "output/callback.stdout").read_text() == Result.stdout


def test_callback_failure_does_not_replace_original_success_exit(monkeypatch, tmp_path):
	prompt = tmp_path / "prompt.md"
	prompt.write_text("finish the task")
	process = _FakeProcess([])

	class Result:
		returncode = 9
		stdout = ""
		stderr = "queue rejected\n"

	monkeypatch.setattr("scripts.coding.run.subprocess.Popen", lambda *args, **kwargs: process)
	monkeypatch.setattr("scripts.coding.run.subprocess.run", lambda *args, **kwargs: Result())
	with pytest.raises(typer.Exit) as error:
		main(prompt, tmp_path / "output", profile="test", cwd=tmp_path, notify_thread="parent-thread")

	assert error.value.exit_code == 0
	receipt = json.loads((tmp_path / "output/callback-receipt.json").read_text())
	assert receipt["status"] == "failed" and receipt["exit_code"] == 9


def test_callback_timeout_does_not_replace_original_failure_exit(monkeypatch, tmp_path):
	prompt = tmp_path / "prompt.md"
	prompt.write_text("finish the task")
	process = _FakeProcess([], returncode=7)

	def callback(*args, **kwargs):
		raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

	monkeypatch.setattr("scripts.coding.run.subprocess.Popen", lambda *args, **kwargs: process)
	monkeypatch.setattr("scripts.coding.run.subprocess.run", callback)
	with pytest.raises(typer.Exit) as error:
		main(prompt, tmp_path / "output", profile="test", cwd=tmp_path, notify_thread="parent-thread")

	assert error.value.exit_code == 7
	receipt = json.loads((tmp_path / "output/callback-receipt.json").read_text())
	assert receipt["status"] == "timeout"
