import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from lib.profiling import submission


def _inputs(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	manifest = tmp_path / "manifest.json"
	manifest.write_text(json.dumps({"hardware_profile": "cpu-test", "command": ["{python}", "-V"]}))
	hardware = tmp_path / "hardware.json"
	hardware.write_text(json.dumps({"profiles": {"cpu-test": {"cpu": 4, "mem": 8, "gpus": 0}}}))
	return manifest, hardware, tmp_path / "output"


def _freeze(root, destination):
	destination.mkdir(parents=True)
	(destination / "source.patch").write_text("", encoding="utf-8")
	return {"commit": "fixture-commit", "files": {}, "tree_sha256": "tree-hash"}


def _submit_task(task_id="task-1"):
	return SimpleNamespace(returncode=0, stdout=json.dumps({"task_id": task_id, "status": "queued"}), stderr="")


def _matching_task(command, output, task_id="task-recovered"):
	return {
		"task_id": task_id,
		"status": "queued",
		"command": shlex.join(["bash", "-lc", command[-1]]),
		"output": str(output.resolve()),
	}


def test_failed_freeze_leaves_preparing_journal_and_retry_uses_new_attempt(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	freeze_calls = []
	submit_calls = []

	def freeze(root, destination):
		freeze_calls.append(destination)
		if len(freeze_calls) == 1:
			raise OSError("fixture preflight failure")
		return _freeze(root, destination)

	def run(command, **kwargs):
		submit_calls.append(command)
		return _submit_task()

	monkeypatch.setattr(submission, "freeze", freeze)
	monkeypatch.setattr(submission.subprocess, "run", run)
	with pytest.raises(OSError, match="fixture preflight failure"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	journal_path, _ = submission._journal_paths(output.resolve())
	first = json.loads(journal_path.read_text())
	assert first["state"] == "preparing"

	task = submission.submit_once(manifest, hardware, output, "/env/bin/python")
	second = json.loads(journal_path.read_text())
	assert task["task_id"] == "task-1"
	assert second["state"] == "submitted"
	assert second["attempt_id"] != first["attempt_id"]
	assert len(freeze_calls) == 2 and len(submit_calls) == 1


def test_lost_submit_ack_recovers_by_exact_attempt_command_and_output(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(submission, "freeze", _freeze)
	calls = []

	def run(command, **kwargs):
		calls.append(command)
		if "submit" in command:
			raise subprocess.TimeoutExpired(command, timeout=120)
		if "tasks" in command:
			journal_path, _ = submission._journal_paths(output.resolve())
			journal = json.loads(journal_path.read_text())
			return SimpleNamespace(
				returncode=0,
				stdout=json.dumps({"queue": [_matching_task(journal["submit_command"], output)]}),
				stderr="",
			)
		pytest.fail(f"Unexpected scheduler command: {command}")

	monkeypatch.setattr(submission.subprocess, "run", run)
	task = submission.submit_once(manifest, hardware, output, "/env/bin/python")
	journal_path, _ = submission._journal_paths(output.resolve())
	journal = json.loads(journal_path.read_text())
	assert task["task_id"] == "task-recovered"
	assert journal["state"] == "submitted" and journal["task_id"] == task["task_id"]
	assert journal["attempt_id"] in journal["shell_command"]
	assert str(output.resolve()) in journal["shell_command"]
	assert sum("submit" in command for command in calls) == 1
	assert sum("tasks" in command for command in calls) == 1


def test_unmatched_uncertain_submission_blocks_every_automatic_resubmit(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(submission, "freeze", _freeze)
	submit_calls = []
	task_listing = {"tasks": []}

	def run(command, **kwargs):
		if "submit" in command:
			submit_calls.append(command)
			raise subprocess.TimeoutExpired(command, timeout=120)
		return SimpleNamespace(returncode=0, stdout=json.dumps(task_listing), stderr="")

	monkeypatch.setattr(submission.subprocess, "run", run)
	with pytest.raises(submission.SubmissionUncertain, match="will not be retried automatically"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	journal_path, _ = submission._journal_paths(output.resolve())
	journal = json.loads(journal_path.read_text())
	assert journal["state"] == "submitting"
	# A same-attempt command with the wrong output is not an acceptable recovery match.
	task_listing["tasks"] = [
		{
			"task_id": "wrong-output",
			"command": ["bash", "-lc", journal["submit_command"][-1]],
			"output": str((tmp_path / "other-output").resolve()),
		}
	]
	with pytest.raises(submission.SubmissionUncertain, match="will not be retried automatically"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	assert len(submit_calls) == 1


def test_submitted_journal_retries_task_ledger_write_without_resubmitting(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(submission, "freeze", _freeze)
	submit_calls = []
	monkeypatch.setattr(
		submission.subprocess,
		"run",
		lambda command, **kwargs: submit_calls.append(command) or _submit_task(),
	)
	real_atomic_json = submission._atomic_json
	failed = False

	def fail_task_ledger_once(path, value):
		nonlocal failed
		if path.name == "task-1.json" and not failed:
			failed = True
			raise OSError("fixture ledger write failure")
		return real_atomic_json(path, value)

	monkeypatch.setattr(submission, "_atomic_json", fail_task_ledger_once)
	with pytest.raises(OSError, match="fixture ledger write failure"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	journal_path, _ = submission._journal_paths(output.resolve())
	assert json.loads(journal_path.read_text())["state"] == "submitted"
	task = submission.submit_once(manifest, hardware, output, "/env/bin/python")
	assert task["task_id"] == "task-1"
	assert (journal_path.parent / "task-1.json").is_file()
	assert len(submit_calls) == 1


def test_known_submitted_task_is_returned_without_freeze_or_scheduler_contact(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	freeze_calls, scheduler_calls = [], []
	monkeypatch.setattr(
		submission, "freeze", lambda root, destination: freeze_calls.append(destination) or _freeze(root, destination)
	)
	monkeypatch.setattr(
		submission.subprocess,
		"run",
		lambda command, **kwargs: scheduler_calls.append(command) or _submit_task("saved-task"),
	)
	first = submission.submit_once(manifest, hardware, output, "/env/bin/python")
	second = submission.submit_once(manifest, hardware, output, "/env/bin/python")
	assert first["task_id"] == second["task_id"] == "saved-task"
	assert len(freeze_calls) == len(scheduler_calls) == 1


def test_changed_frozen_call_is_rejected_for_a_reserved_output(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	monkeypatch.setattr(submission, "freeze", _freeze)
	scheduler_calls = []
	monkeypatch.setattr(
		submission.subprocess,
		"run",
		lambda command, **kwargs: scheduler_calls.append(command) or _submit_task(),
	)
	submission.submit_once(manifest, hardware, output, "/env/bin/python")
	hardware.write_text(json.dumps({"profiles": {"cpu-test": {"cpu": 8, "mem": 8, "gpus": 0}}}))
	with pytest.raises(submission.SubmissionConflict, match="different manifest, hardware, or Python"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	assert len(scheduler_calls) == 1


def test_concurrent_calls_for_one_output_share_one_submission(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	freeze_started, allow_freeze = Event(), Event()
	freeze_guard, freeze_calls, submit_calls = Lock(), [], []

	def freeze(root, destination):
		with freeze_guard:
			freeze_calls.append(destination)
			first = len(freeze_calls) == 1
		if first:
			freeze_started.set()
			if not allow_freeze.wait(timeout=5):
				raise TimeoutError("fixture synchronization timed out")
		return _freeze(root, destination)

	def run(command, **kwargs):
		submit_calls.append(command)
		return _submit_task("shared-task")

	monkeypatch.setattr(submission, "freeze", freeze)
	monkeypatch.setattr(submission.subprocess, "run", run)
	with ThreadPoolExecutor(max_workers=2) as pool:
		first = pool.submit(submission.submit_once, manifest, hardware, output, "/env/bin/python")
		assert freeze_started.wait(timeout=5)
		second = pool.submit(submission.submit_once, manifest, hardware, output, "/env/bin/python")
		allow_freeze.set()
		first_task, second_task = first.result(timeout=10), second.result(timeout=10)
	assert first_task["task_id"] == second_task["task_id"] == "shared-task"
	assert len(freeze_calls) == len(submit_calls) == 1


def test_legacy_queued_reservation_is_not_ignored_during_upgrade(tmp_path, monkeypatch):
	manifest, hardware, output = _inputs(tmp_path, monkeypatch)
	journal, _ = submission._journal_paths(output)
	journal.parent.mkdir(parents=True)
	legacy = journal.with_name(journal.name.replace(".submission.json", ".reservation.json"))
	legacy.write_text('{"state":"submitted","task_id":"old-running-task"}')
	with pytest.raises(submission.SubmissionUncertain, match="old-running-task"):
		submission.submit_once(manifest, hardware, output, "/env/bin/python")
	assert not journal.exists()
