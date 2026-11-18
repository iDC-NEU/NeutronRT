"""Exactly-once submission journal for frozen Neu Box manifests.

The scheduler API does not provide an idempotency key. This module therefore
persists an attempt UUID and the exact command before submitting. If the submit
acknowledgement is lost, it reconciles that UUID and output path against the
scheduler task list and refuses to submit again when the result is uncertain.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


class SubmissionConflict(RuntimeError):
	"""The same output path was reused with changed frozen inputs."""


class SubmissionUncertain(RuntimeError):
	"""A scheduler submission may exist, so it is unsafe to submit again."""


def digest(path: Path) -> str:
	with path.open("rb") as stream:
		return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(root: Path, destination: Path) -> dict[str, Any]:
	"""Use the source snapshot implementation supplied by the parent checkout."""
	from lib.profiling.snapshot import freeze as freeze_source

	return freeze_source(root, destination)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
		) as stream:
			temporary = Path(stream.name)
			json.dump(value, stream, indent=2, sort_keys=True)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
		try:
			directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
		except OSError:
			return
		try:
			os.fsync(directory)
		finally:
			os.close(directory)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def _read_journal(path: Path) -> dict[str, Any]:
	try:
		value = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		raise SubmissionUncertain(f"Cannot read the output submission journal {path}: {error}") from error
	if not isinstance(value, dict):
		raise SubmissionUncertain(f"Output submission journal must be an object: {path}")
	return value


def _request_identity(manifest_path: Path, hardware_path: Path, output: Path, python: str) -> dict[str, str]:
	return {
		"manifest_path": str(manifest_path.resolve()),
		"manifest_sha256": digest(manifest_path),
		"hardware_path": str(hardware_path.resolve()),
		"hardware_sha256": digest(hardware_path),
		"output": str(output.resolve()),
		"python": str(python),
	}


def _journal_paths(output: Path) -> tuple[Path, Path]:
	ledger = Path.cwd() / "logs" / "task-ledger"
	key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
	return ledger / f"{key}.submission.json", ledger / f"{key}.lock"


def _resource_profile(manifest: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
	try:
		profile = hardware["profiles"][manifest["hardware_profile"]]
		cpu, memory, gpus = profile["cpu"], profile["mem"], profile["gpus"]
	except (KeyError, TypeError) as error:
		raise ValueError("Manifest must name a valid frozen hardware profile") from error
	if type(cpu) is not int or type(memory) is not int or type(gpus) is not int or cpu < 1 or memory < 1 or gpus < 0:
		raise ValueError("Hardware profile must have bounded positive CPU/RAM and nonnegative GPU count")
	return dict(profile)


def _save_submitted(
	journal_path: Path,
	journal: dict[str, Any],
	task: dict[str, Any],
) -> dict[str, Any]:
	task_id = task.get("task_id")
	if not isinstance(task_id, str) or not task_id.strip():
		raise SubmissionUncertain("Scheduler acknowledgement omitted a usable task_id")
	updated = {
		**journal,
		"state": "submitted",
		"task_id": task_id,
		"task": task,
		"updated_unix": time.time(),
	}
	# Persist acceptance before any secondary task-ledger write. A retry can
	# reconstruct the task ID from this journal if the latter write fails.
	_atomic_json(journal_path, updated)
	_write_task_ledger(journal_path, updated)
	return dict(task or {})


def _write_task_ledger(journal_path: Path, journal: dict[str, Any]) -> None:
	task_id = journal.get("task_id")
	if (
		not isinstance(task_id, str)
		or not task_id.strip()
		or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in task_id)
	):
		raise SubmissionUncertain("Cannot write task ledger for an invalid task_id")
	ledger_path = journal_path.parent / f"{task_id}.json"
	if ledger_path.exists():
		previous = _read_journal(ledger_path)
		previous_task = previous.get("task")
		if not isinstance(previous_task, dict):
			previous_task = {}
		if previous_task.get("task_id") != task_id or previous.get("output") != journal.get("output"):
			raise SubmissionConflict(f"Task ledger entry already belongs to another output: {ledger_path}")
		return
	payload = {
		"task": journal.get("task") or {"task_id": task_id},
		"profile": journal["profile"],
		"manifest": journal["manifest"],
		"output": journal["output"],
		"snapshot": journal["snapshot"],
		"attempt_id": journal["attempt_id"],
		"journal": str(journal_path.resolve()),
	}
	_atomic_json(ledger_path, payload)


def _task_rows(payload: Any) -> list[dict[str, Any]]:
	if isinstance(payload, list):
		rows = payload
	elif isinstance(payload, dict) and isinstance(payload.get("queue"), list):
		rows = payload["queue"]
	elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
		rows = payload["tasks"]
	elif isinstance(payload, dict) and payload.get("task_id"):
		rows = [payload]
	else:
		raise SubmissionUncertain("Scheduler task listing omitted a tasks array")
	return [row for row in rows if isinstance(row, dict)]


def _command_matches(task: dict[str, Any], journal: dict[str, Any]) -> bool:
	value = task.get("command", task.get("cmd", task.get("command_line")))
	if isinstance(value, str):
		return value in journal["command_strings"]
	if isinstance(value, list) and all(isinstance(item, str) for item in value):
		return value in journal["command_argv"]
	return False


def _task_command_text(task: dict[str, Any]) -> str:
	value = task.get("command", task.get("cmd", task.get("command_line")))
	if isinstance(value, str):
		return value
	if isinstance(value, list) and all(isinstance(item, str) for item in value):
		return shlex.join(value)
	return ""


def _output_matches(task: dict[str, Any], output: str) -> bool:
	for key in ("output", "output_path", "output_dir"):
		value = task.get(key)
		if value is not None and (not isinstance(value, str) or str(Path(value).expanduser().resolve()) != output):
			return False
	return True


def _reconcile(journal_path: Path, journal: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
	try:
		result = subprocess.run(
			["neu-sbox", "--json", "tasks"],
			capture_output=True,
			text=True,
			check=False,
			timeout=timeout,
		)
	except (OSError, subprocess.TimeoutExpired) as error:
		raise SubmissionUncertain(
			f"Could not reconcile submission attempt {journal['attempt_id']}; it will not be resubmitted: {error}"
		) from error
	if result.returncode != 0:
		raise SubmissionUncertain(
			f"Could not reconcile submission attempt {journal['attempt_id']} (tasks returned {result.returncode}); "
			"it will not be resubmitted"
		)
	try:
		payload = json.loads(result.stdout)
	except json.JSONDecodeError as error:
		raise SubmissionUncertain("Scheduler task listing returned invalid JSON; refusing to resubmit") from error
	matches_by_id: dict[str, dict[str, Any]] = {}
	for task in _task_rows(payload):
		task_id = task.get("task_id")
		if (
			isinstance(task_id, str)
			and task_id.strip()
			and _command_matches(task, journal)
			and _output_matches(task, journal["output"])
			and journal["attempt_id"] in _task_command_text(task)
		):
			matches_by_id[task_id] = task
	if not matches_by_id:
		journal["last_reconcile_unix"] = time.time()
		journal["last_reconcile_result"] = "no exact command/output/attempt match"
		_atomic_json(journal_path, journal)
		raise SubmissionUncertain(
			f"No exact scheduler task match for attempt {journal['attempt_id']} and output {journal['output']}; "
			"submission remains uncertain and will not be retried automatically"
		)
	if len(matches_by_id) != 1:
		journal["last_reconcile_unix"] = time.time()
		journal["last_reconcile_result"] = f"ambiguous task IDs: {sorted(matches_by_id)}"
		_atomic_json(journal_path, journal)
		raise SubmissionUncertain(
			f"Multiple scheduler tasks match attempt {journal['attempt_id']}; refusing to choose or resubmit"
		)
	return _save_submitted(journal_path, journal, next(iter(matches_by_id.values())))


def _prepare_attempt(
	*,
	journal_path: Path,
	journal: dict[str, Any],
	manifest_path: Path,
	hardware_path: Path,
	manifest: dict[str, Any],
	profile: dict[str, Any],
	python: str,
) -> dict[str, Any]:
	attempt_id = uuid4().hex
	snapshot_path = (Path.cwd() / "logs" / "submissions" / attempt_id).resolve()
	source_path = snapshot_path / "source"
	journal = {
		**journal,
		"state": "preparing",
		"attempt_id": attempt_id,
		"snapshot": str(snapshot_path),
		"updated_unix": time.time(),
	}
	_atomic_json(journal_path, journal)

	source = freeze(Path.cwd(), source_path)
	frozen_manifest = {
		**manifest,
		"hardware": profile,
		"source_snapshot": source,
		"submission_attempt_id": attempt_id,
	}
	frozen_manifest["expected_inputs_sha256"] = {name: digest(Path(name)) for name in manifest.get("hash_files", [])}
	frozen_path = snapshot_path / "manifest.json"
	_atomic_json(frozen_path, frozen_manifest)
	manifest_sha256 = digest(frozen_path)
	command = [
		python,
		"-m",
		"scripts.evaluate.run_manifest",
		str(frozen_path),
		"--output",
		journal["output"],
		"--expected-manifest-sha256",
		manifest_sha256,
	]
	shell_command = (
		f"export NEUTRONRT_SUBMISSION_ATTEMPT={attempt_id}; "
		f"cd {shlex.quote(str(source_path))} && exec {shlex.join(command)}"
	)
	submit_command = [
		"neu-sbox",
		"--json",
		"submit",
		"--cpu",
		str(profile["cpu"]),
		"--mem",
		str(profile["mem"]),
		"--device-num",
		str(profile["gpus"]),
		"--",
		"bash",
		"-lc",
		shell_command,
	]
	if digest(manifest_path) != journal["request"]["manifest_sha256"]:
		raise SubmissionConflict("Manifest changed during submission preflight")
	if digest(hardware_path) != journal["request"]["hardware_sha256"]:
		raise SubmissionConflict("Hardware profile changed during submission preflight")
	journal = {
		**journal,
		"state": "submitting",
		"manifest": frozen_manifest,
		"profile": profile,
		"frozen_manifest": str(frozen_path),
		"frozen_manifest_sha256": manifest_sha256,
		"worker_command": command,
		"shell_command": shell_command,
		"submit_command": submit_command,
		"command_strings": [
			shlex.join(command),
			shell_command,
			shlex.join(["bash", "-lc", shell_command]),
			shlex.join(submit_command),
		],
		"command_argv": [command, ["bash", "-lc", shell_command], submit_command],
		"updated_unix": time.time(),
	}
	_atomic_json(journal_path, journal)
	return _submit_or_reconcile(journal_path, journal)


def _submit_or_reconcile(journal_path: Path, journal: dict[str, Any]) -> dict[str, Any]:
	try:
		result = subprocess.run(
			journal["submit_command"],
			capture_output=True,
			text=True,
			check=False,
			timeout=120,
		)
	except (OSError, subprocess.TimeoutExpired):
		return _reconcile(journal_path, journal, timeout=30.0)
	if result.returncode == 0:
		try:
			task = json.loads(result.stdout)
		except json.JSONDecodeError:
			return _reconcile(journal_path, journal)
		if isinstance(task, dict) and isinstance(task.get("task_id"), str) and task["task_id"].strip():
			return _save_submitted(journal_path, journal, task)
	return _reconcile(journal_path, journal)


def submit_once(manifest_path: Path, hardware_path: Path, output: Path, python: str) -> dict[str, Any]:
	"""Freeze and submit a manifest once per output path, recovering lost acks safely."""
	manifest_path = Path(manifest_path).resolve()
	hardware_path = Path(hardware_path).resolve()
	output = Path(output).resolve()
	if not isinstance(python, str) or not python:
		raise ValueError("A Python executable path is required")
	request = _request_identity(manifest_path, hardware_path, output, python)
	journal_path, lock_path = _journal_paths(output)
	lock_path.parent.mkdir(parents=True, exist_ok=True)
	with lock_path.open("a+") as lock:
		fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
		if journal_path.exists():
			journal = _read_journal(journal_path)
			if journal.get("request") != request:
				raise SubmissionConflict(
					"This output path is already reserved for different manifest, hardware, or Python inputs"
				)
			state = journal.get("state")
			if state == "submitted":
				task = (
					journal.get("task")
					if isinstance(journal.get("task"), dict)
					else {"task_id": journal.get("task_id")}
				)
				_write_task_ledger(journal_path, journal)
				return dict(task or {})
			if state == "submitting":
				return _reconcile(journal_path, journal)
			if state != "preparing":
				raise SubmissionUncertain(f"Unknown submission journal state {state!r}; refusing to submit")
		else:
			legacy = journal_path.with_name(journal_path.name.replace(".submission.json", ".reservation.json"))
			if legacy.exists():
				record = _read_journal(legacy)
				raise SubmissionUncertain(
					f"Legacy output reservation already exists (task {record.get('task_id', 'unknown')}); "
					"inspect that task rather than submitting a duplicate"
				)
			if output.exists():
				raise FileExistsError(f"Output directory already exists: {output}")
			journal = {
				"schema": 1,
				"state": "preparing",
				"request": request,
				"output": str(output),
				"created_unix": time.time(),
				"updated_unix": time.time(),
			}
			_atomic_json(journal_path, journal)

		# A previous preparation stopped before contacting the scheduler. Replace
		# its attempt UUID and snapshot so incomplete preflight artifacts are not reused.
		if journal.get("state") == "preparing":
			try:
				manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
				hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
				profile = _resource_profile(manifest, hardware)
				for name in manifest.get("required_files", []):
					if not Path(name).is_file():
						raise ValueError(f"Required input file does not exist: {name}")
			except (OSError, json.JSONDecodeError, ValueError) as error:
				# No scheduler call has happened yet, so retaining preparing is safe.
				raise ValueError(f"Submission preflight failed before scheduler contact: {error}") from error
			if output.exists():
				raise FileExistsError(f"Output directory already exists: {output}")
			return _prepare_attempt(
				journal_path=journal_path,
				journal=journal,
				manifest_path=manifest_path,
				hardware_path=hardware_path,
				manifest=manifest,
				profile=profile,
				python=python,
			)
		raise SubmissionUncertain("Submission state changed unexpectedly; no new scheduler request was sent")
