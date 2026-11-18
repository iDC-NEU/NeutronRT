"""Run a persistent profile-based Codex session and consume its event stream."""

import fcntl
import json
import os
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

import typer


def command(profile: str, cwd: Path, result: Path, resume: str | None) -> list[str]:
	args = [
		"codex",
		"--profile",
		profile,
		"--sandbox",
		"workspace-write",
		"exec",
		"--json",
		"--output-last-message",
		str(result.resolve()),
	]
	if resume:
		args += ["resume", resume, "-"]
	else:
		args += ["-C", str(cwd.resolve()), "-"]
	return ["timeout", "--foreground", "--kill-after=30s", "1h", *args]


def _write_state(path: Path, state: dict) -> None:
	"""Publish a complete session state file without exposing partial JSON."""
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
		) as stream:
			temporary = Path(stream.name)
			json.dump(state, stream, indent=2)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def _stop(process: subprocess.Popen) -> None:
	if process.poll() is not None:
		return
	process.terminate()
	try:
		process.wait(timeout=5)
	except subprocess.TimeoutExpired:
		process.kill()
		try:
			process.wait(timeout=5)
		except subprocess.TimeoutExpired:
			# The original task result is already known; never wait forever on a
			# broken child process while attempting cleanup.
			return


def _write_callback(output: Path, notify_thread: str | None, state: dict) -> dict:
	receipt: dict[str, Any] = {
		"requested": bool(notify_thread),
		"thread_id": notify_thread,
		"attempted": False,
		"delivery_verified": False,
	}
	if not notify_thread:
		receipt["status"] = "not_requested"
	elif notify_thread == state.get("thread_id"):
		receipt.update(status="skipped_recursive_notification", reason="notify_thread_matches_session_thread")
	else:
		payload = {
			"run": str(output.resolve()),
			"state": state.get("state"),
			"returncode": state.get("returncode"),
			"timed_out": bool(state.get("timed_out", False)),
			"thread_id": state.get("thread_id"),
			"error": state.get("error"),
		}
		message = "CODEX_RUN_COMPLETE " + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
		message = message[:1800]
		try:
			result = subprocess.run(
				["codex", "queue", "--resume", "--thread", notify_thread, "--message", message],
				capture_output=True,
				text=True,
				timeout=30,
				check=False,
			)
			receipt.update(
				attempted=True,
				status="queued" if result.returncode == 0 else "failed",
				exit_code=result.returncode,
				queue_warning="warning" in (result.stderr or "").lower() or "warn" in (result.stderr or "").lower(),
			)
			try:
				(output / "callback.stdout").write_text(result.stdout or "")
				(output / "callback.stderr").write_text(result.stderr or "")
			except OSError as error:
				receipt["output_error"] = f"{type(error).__name__}: {error}"
		except subprocess.TimeoutExpired as error:
			receipt.update(attempted=True, status="timeout", error=str(error))
		except OSError as error:
			receipt.update(attempted=True, status="failed", error=f"{type(error).__name__}: {error}")
		except BaseException as error:
			receipt.update(attempted=True, status="failed", error=f"{type(error).__name__}: {error}")
	with suppress(OSError):
		(output / "callback-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
	return receipt


def main(
	prompt_file: Annotated[Path, typer.Argument(exists=True)],
	output: Annotated[Path, typer.Option("--output")],
	profile: str = "lab",
	cwd: Path = Path("."),
	resume: str | None = None,
	notify_thread: str | None = None,
):
	"""Launch in a harness background terminal; do not use --ephemeral or shell &."""
	output.mkdir(parents=True, exist_ok=False)
	locks = []
	state: dict[str, Any] = {"profile": profile, "cwd": str(cwd.resolve()), "thread_id": resume, "state": "running"}
	state_path = output / "session.json"
	try:
		if profile == "lab":
			lock_root = Path("/tmp/neutronrt-lab-slots")
			lock_root.mkdir(exist_ok=True)
			for slot in range(2):
				lock = (lock_root / str(slot)).open("a")
				try:
					fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
					locks.append(lock)
					break
				except BlockingIOError:
					lock.close()
			if not locks:
				state.update(state="failed", error="Two lab sessions are already running")
				_write_state(state_path, state)
				raise RuntimeError("Two lab sessions are already running")
		_write_state(state_path, state)
		args = command(profile, cwd, output / "result.md", resume)
		process = None
		failure_event = None
		code = None
		try:
			with (
				prompt_file.open() as prompt,
				(output / "stderr.log").open("w") as errors,
				(output / "events.jsonl").open("w") as events,
			):
				process = subprocess.Popen(args, stdin=prompt, stdout=subprocess.PIPE, stderr=errors, text=True)
				assert process.stdout is not None
				for line in process.stdout:
					events.write(line)
					events.flush()
					try:
						event = json.loads(line)
					except json.JSONDecodeError:
						continue
					if event.get("type") == "thread.started":
						state["thread_id"] = event["thread_id"]
					if event.get("type") in ("turn.failed", "error"):
						failure_event = event
						state.update(state="failed", error=event.get("message") or event.get("error") or event)
					if event.get("type") in ("thread.started", "turn.completed", "turn.failed", "error"):
						_write_state(state_path, state)
						typer.echo(line.rstrip())
				try:
					code = process.wait(timeout=5)
				except subprocess.TimeoutExpired:
					state.update(timed_out=True, error="Codex session exceeded the one-hour timeout")
					_stop(process)
					code = 124
		finally:
			if process is not None and process.poll() is None:
				_stop(process)
			if process is not None and process.stdout is not None:
				close = getattr(process.stdout, "close", None)
				if close is not None:
					close()
		if failure_event is not None and code == 0:
			code = 1
		if code is None:
			code = 1
		if code == 124 and not state.get("timed_out"):
			state.update(timed_out=True, error="Codex session exceeded the one-hour timeout")
		state.update(state="completed" if code == 0 else "failed", returncode=code)
		_write_state(state_path, state)
		raise typer.Exit(code)
	except BaseException as error:
		if state.get("state") == "running":
			state.update(state="failed", error=f"{type(error).__name__}: {error}")
			_write_state(state_path, state)
		raise
	finally:
		if state.get("state") != "running":
			_write_callback(output, notify_thread, state)
		for lock in locks:
			lock.close()


if __name__ == "__main__":
	typer.run(main)
