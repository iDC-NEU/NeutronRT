"""Bound a task watch and queue its result back to the owning Codex session."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Annotated, Optional

import typer

_DEFAULT_SECONDS = 3300
_MAX_SECONDS = 3600
_WORKER_GRACE_SECONDS = 5
_CALLBACK_SECONDS = 30


def _atomic_write(path: Path, value: dict) -> None:
	temporary: Optional[Path] = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
		) as stream:
			temporary = Path(stream.name)
			json.dump(value, stream, indent=2)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary is not None:
			with suppress(FileNotFoundError):
				temporary.unlink()


def _worker_command(task_ids: list[str], seconds: int) -> list[str]:
	worker = os.environ.get("NEUTRONRT_WATCH_WORKER")
	command = [sys.executable, worker] if worker else [sys.executable, "-m", "scripts.neubox.tasks"]
	options = ["--watch", "--json", "--seconds", str(seconds)]
	for task in task_ids:
		options.extend(["--task-id", task])
	return [*command, *options]


def _terminate_group(process: subprocess.Popen, *, grace: float = _WORKER_GRACE_SECONDS) -> None:
	if process.poll() is not None:
		return
	try:
		os.killpg(process.pid, signal.SIGTERM)
	except (ProcessLookupError, PermissionError, OSError):
		with suppress(ProcessLookupError, PermissionError, OSError):
			process.terminate()
	try:
		process.wait(timeout=max(0.1, grace))
	except subprocess.TimeoutExpired:
		try:
			os.killpg(process.pid, signal.SIGKILL)
		except (ProcessLookupError, PermissionError, OSError):
			with suppress(ProcessLookupError, PermissionError, OSError):
				process.kill()
		with suppress(subprocess.TimeoutExpired):
			process.wait(timeout=max(0.1, grace))


def _callback_timeout() -> float:
	value = os.environ.get("NEUTRONRT_WATCH_QUEUE_TIMEOUT")
	if value is None:
		return _CALLBACK_SECONDS
	try:
		return max(0.01, min(_CALLBACK_SECONDS, float(value)))
	except ValueError:
		return _CALLBACK_SECONDS


def _queue_callback(receipt_path: Path, output: Path, task_ids: list[str], thread: str, receipt: dict) -> None:
	receipt["callback_attempted"] = True
	receipt["callback_status"] = "starting"
	with suppress(OSError):
		_atomic_write(receipt_path, receipt)
	message = (
		f"Automated neu-sbox watch callback: tasks {', '.join(task_ids)}; monitor exit {receipt['monitor_exit']}. "
		f"Log: {output.resolve() / 'watch.jsonl'}. Check task health and experiment progress now. "
		"If tasks remain active, renew one bounded watch with this callback. Do not resubmit experiments."
	)
	message = message[:1800]
	command = [
		os.environ.get("NEUTRONRT_CODEX_EXECUTABLE", "codex"),
		"queue",
		"--resume",
		"--thread",
		thread,
		"--message",
		message,
	]
	environment = {**os.environ, "NEUTRONRT_WATCH_RECEIPT": str(receipt_path.resolve())}
	started = monotonic()
	try:
		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			env=environment,
			timeout=_callback_timeout(),
			check=False,
		)
		diagnostic = f"{result.stdout}\n{result.stderr}"
		receipt.update(
			queue_exit=result.returncode,
			callback_status="queued" if result.returncode == 0 else "failed",
			queue_warning="warning" in diagnostic.lower() or "active writer" in diagnostic.lower(),
			resume_verified=False,
			callback_seconds=monotonic() - started,
		)
		with (output / "callback.log").open("w", encoding="utf-8") as log:
			log.write(diagnostic)
	except subprocess.TimeoutExpired as error:
		receipt.update(
			queue_exit=None,
			callback_status="timeout",
			resume_verified=False,
			callback_seconds=monotonic() - started,
			callback_error=str(error),
		)
		with (output / "callback.log").open("w", encoding="utf-8") as log:
			log.write(f"queue callback timed out: {error}\n")
	except BaseException as error:
		receipt.update(
			queue_exit=None,
			callback_status="failed",
			resume_verified=False,
			callback_seconds=monotonic() - started,
			callback_error=f"{type(error).__name__}: {error}",
		)
		with suppress(OSError):
			(output / "callback.log").write_text(receipt["callback_error"] + "\n", encoding="utf-8")


def _monitor(task_ids: list[str], output: Path, thread: str, seconds: int) -> int:
	watch_log = output / "watch.jsonl"
	receipt_path = output / "receipt.json"
	worker: Optional[subprocess.Popen] = None
	received_signal: Optional[int] = None
	received_signals: list[int] = []
	monitor_exit: Optional[int] = None
	monitor_status = "unknown"
	worker_started = monotonic()
	exception = None

	def handle_signal(signum, _frame) -> None:
		nonlocal received_signal
		received_signals.append(signum)
		if received_signal is None:
			received_signal = signum
		if worker is not None:
			_terminate_group(worker)

	previous_handlers = {}
	for signum in (signal.SIGTERM, signal.SIGINT):
		previous_handlers[signum] = signal.signal(signum, handle_signal)
	try:
		try:
			with watch_log.open("w", encoding="utf-8") as log:
				worker = subprocess.Popen(
					_worker_command(task_ids, seconds),
					stdout=log,
					stderr=subprocess.STDOUT,
					start_new_session=True,
				)
				try:
					worker.wait(timeout=seconds + _WORKER_GRACE_SECONDS)
				except subprocess.TimeoutExpired:
					monitor_status = "worker_timeout"
					_terminate_group(worker)
					monitor_exit = 124
				else:
					monitor_exit = worker.returncode
					monitor_status = "worker_completed" if monitor_exit == 0 else "worker_failed"
			if received_signal is not None:
				monitor_status = "interrupted"
				monitor_exit = worker.returncode if worker is not None else None
		except BaseException as error:
			monitor_status = "wrapper_exception"
			monitor_exit = None
			exception = f"{type(error).__name__}: {error}"
		if monitor_exit == 124 and monitor_status == "worker_failed":
			monitor_status = "worker_timeout"
		original_exit = (
			128 + received_signal if received_signal is not None else monitor_exit if monitor_exit is not None else 1
		)
		receipt = {
			"task_ids": task_ids,
			"thread": thread,
			"monitor_exit": monitor_exit,
			"wrapper_exit": original_exit,
			"monitor_status": monitor_status,
			"signal": received_signal,
			"signals": received_signals,
			"exception": exception,
			"elapsed_seconds": monotonic() - worker_started,
			"outer_deadline_seconds": seconds + _WORKER_GRACE_SECONDS,
			"headroom_warning": seconds >= _MAX_SECONDS,
			"callback_attempted": False,
			"callback_status": "pending",
			"resume_verified": False,
		}
		# Save the worker outcome before queueing; acceptance does not prove delivery.
		with suppress(OSError):
			_atomic_write(receipt_path, receipt)
		try:
			_queue_callback(receipt_path, output, task_ids, thread, receipt)
		except BaseException as error:
			receipt.update(
				callback_attempted=True,
				callback_status="failed",
				resume_verified=False,
				callback_error=f"{type(error).__name__}: {error}",
			)
		with suppress(OSError):
			_atomic_write(receipt_path, receipt)
		# Keep signal handlers installed until callback and receipt work finishes.
		if received_signal is not None:
			original_exit = 128 + received_signal
			receipt.update(signal=received_signal, signals=received_signals, wrapper_exit=original_exit)
			with suppress(OSError):
				_atomic_write(receipt_path, receipt)
	finally:
		for signum, handler in previous_handlers.items():
			with suppress(ValueError):
				signal.signal(signum, handler)
	return original_exit


def main(
	task_ids: Annotated[list[str], typer.Option("--task-id")],
	output: Annotated[Path, typer.Option("--output")],
	thread: Annotated[str | None, typer.Option("--thread")] = None,
	seconds: Annotated[int, typer.Option(min=1, max=_MAX_SECONDS)] = _DEFAULT_SECONDS,
):
	"""Run a background watch and queue exactly one bounded completion message."""
	thread = thread or os.environ.get("CODEX_THREAD_ID")
	if not thread:
		raise typer.BadParameter("Provide --thread or CODEX_THREAD_ID")
	output.mkdir(parents=True, exist_ok=False)
	if seconds >= _MAX_SECONDS:
		typer.echo("warning: a 3600-second watch leaves only bounded cleanup/callback headroom", err=True)
	code = _monitor(task_ids, output, thread, seconds)
	typer.echo(f"Monitor exited {code}; callback receipt: {output / 'receipt.json'}")
	raise typer.Exit(code)


if __name__ == "__main__":
	typer.run(main)
