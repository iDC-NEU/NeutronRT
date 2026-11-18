import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.neubox import timer, watchdog
from scripts.neubox.session import SessionOwner


def owner(pid=None):
	pid = os.getpid() if pid is None else pid
	fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
	return SessionOwner(pid, fields[19], "test-thread")


def test_discovery_ignores_history_tracks_new_tasks_and_saves_full_logs(tmp_path, monkeypatch):
	(tmp_path / "task-logs").mkdir()
	calls = []

	def query(host, command, *args):
		calls.append((host, command, args))
		if command == "tasks":
			return {
				"queue": [
					{"task_id": "old", "user_id": "lhr", "status": "completed"},
					{"task_id": "new", "user_id": "lhr", "status": "running"},
					{"task_id": "other", "user_id": "someone", "status": "running"},
				]
			}
		return {"task_id": "new", "status": "running", "log": "x" * 5000}

	monkeypatch.setattr(watchdog, "query", query)
	tasks = {}
	assert watchdog.scan(["local", "peer"], "lhr", tasks, tmp_path) == []
	assert set(tasks) == {"local:new", "peer:new"}
	assert len(tasks["peer:new"]["log_tail"]) == 2000
	assert len(json.loads((tmp_path / "task-logs/peer-new.json").read_text())["log"]) == 5000
	assert len(calls) == 4


def test_scheduler_error_never_means_all_done(tmp_path, monkeypatch):
	monkeypatch.setattr(watchdog, "query", lambda *_: (_ for _ in ()).throw(OSError("offline")))
	monkeypatch.setattr(watchdog, "notify", lambda *_: pytest.fail("False all-done callback"))
	watchdog.monitor(tmp_path, "test-thread", ["local"], "lhr", 0.03, 0.01, owner())
	state = json.loads((tmp_path / "state.json").read_text())
	assert state["errors"] and state["status"] == "window_elapsed"


def test_completion_receipt_once_and_resume_does_not_notify_twice(tmp_path, monkeypatch):
	monkeypatch.setattr(watchdog, "query", lambda *_: {"queue": []})
	calls = []
	monkeypatch.setattr(watchdog, "notify", lambda *args: calls.append(args) or {"queue_exit": 0})
	for _ in range(2):
		watchdog.monitor(tmp_path, "test-thread", ["local"], "lhr", 0.1, 0.01, owner())
	assert len(calls) == 1
	assert json.loads((tmp_path / "state.json").read_text())["status"] == "all_done"


def test_timer_queues_only_after_interval(tmp_path, monkeypatch):
	calls = []
	monkeypatch.setattr(timer, "notify", lambda *args: calls.append(args) or {"queue_exit": 0})
	assert timer.run(tmp_path / "timer", "test-thread", 0.01, tmp_path / "state.json", owner())["queue_exit"] == 0
	assert len(calls) == 1


def test_owner_exit_stops_wait_and_timer_without_touching_independent_task(tmp_path, monkeypatch):
	process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.1)"])
	independent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
	try:
		bound = owner(process.pid)
		monkeypatch.setattr(timer, "notify", lambda *_: pytest.fail("Owner is gone"))
		assert timer.run(tmp_path / "timer", "test-thread", 2, tmp_path / "state.json", bound)["owner_gone"]
		assert not bound.alive()
		assert independent.poll() is None
		assert not SessionOwner(os.getpid(), "wrong-start-time", "test").alive()
	finally:
		process.wait(timeout=2)
		independent.terminate()
		independent.wait(timeout=2)
