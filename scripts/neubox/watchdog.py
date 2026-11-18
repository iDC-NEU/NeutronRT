"""Discover a user's tasks on local/SSH schedulers and notify when all finish."""

import fcntl
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from scripts.neubox.notify import notify
from scripts.neubox.session import SessionOwner
from scripts.neubox.tasks import TERMINAL_STATUSES
from scripts.neubox.watch import _atomic_write


def query(host: str, *arguments: str) -> dict:
	command = ["neu-sbox", "--json", *arguments]
	if host != "local":
		command = ["ssh", "-oBatchMode=yes", "-oConnectTimeout=10", host, shlex.join(command)]
	result = subprocess.run(command, capture_output=True, text=True, timeout=40, check=True)
	payload = json.loads(result.stdout)
	if not isinstance(payload, dict):
		raise ValueError("Scheduler response is not an object")
	return payload


def scan(hosts: list[str], user: str, tasks: dict, output: Path) -> list[dict]:
	"""Track new active tasks; never interpret a query failure as completion."""
	errors = []
	for host in hosts:
		try:
			payload = query(host, "tasks")
			rows = payload.get("queue", payload.get("tasks"))
			if not isinstance(rows, list):
				raise ValueError("Scheduler omitted queue/tasks array")
			for row in rows:
				if user != "*" and row.get("user_id") != user:
					continue
				key = f"{host}:{row['task_id']}"
				if key not in tasks and row.get("status") not in TERMINAL_STATUSES:
					tasks[key] = {"host": host, "task_id": row["task_id"], "status": row.get("status")}
			for task in [item for item in tasks.values() if item["host"] == host]:
				if task.get("detail_saved") and task["status"] in TERMINAL_STATUSES:
					continue
				detail = query(host, "result", task["task_id"])
				if detail.get("task_id") != task["task_id"] or not isinstance(detail.get("status"), str):
					raise ValueError("Task result missing identity/status")
				task.update(
					{
						key: detail.get(key)
						for key in ("status", "result", "cpu", "mem", "devices", "started_at", "finished_at")
					}
				)
				task["checked_unix"] = time.time()
				task["log_tail"] = str(detail.get("log", ""))[-2000:]
				# Keep the full latest worker result/log, not merely the displayed tail.
				filename = f"{host.replace('/', '_')}-{task['task_id']}.json"
				_atomic_write(output / "task-logs" / filename, detail)
				task["detail_saved"] = True
		except Exception as error:
			errors.append({"host": host, "error": f"{type(error).__name__}: {error}"})
	return errors


def monitor(
	output: Path, thread: str, hosts: list[str], user: str, seconds: float, interval: float, owner: SessionOwner
) -> None:
	output.mkdir(parents=True, exist_ok=True)
	(output / "task-logs").mkdir(exist_ok=True)
	with (output / "watchdog.lock").open("a") as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		state_path = output / "state.json"
		state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {"tasks": {}}
		identity = {"hosts": hosts, "user": user, "thread": thread}
		if state.get("identity", identity) != identity:
			raise ValueError("Watchdog identity changed; choose a new output directory")
		state["identity"] = identity
		state["owner"] = owner.record()
		stop = threading.Event()
		previous_handlers = {}
		for signum in (signal.SIGINT, signal.SIGTERM):
			previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
		deadline = time.monotonic() + seconds
		try:
			while owner.alive() and not stop.is_set() and time.monotonic() < deadline:
				errors = scan(hosts, user, state["tasks"], output)
				active = [key for key, task in state["tasks"].items() if task["status"] not in TERMINAL_STATUSES]
				state.update(errors=errors, active=active, checked_unix=time.time(), status="watching")
				if active:
					state["completion_notified"] = False
				_atomic_write(state_path, state)
				with (output / "events.jsonl").open("a") as stream:
					stream.write(
						json.dumps(
							{
								"unix": time.time(),
								"active": active,
								"errors": errors,
								"statuses": {key: task["status"] for key, task in state["tasks"].items()},
							}
						)
						+ "\n"
					)
				if owner.alive() and not errors and not active:
					if not state.get("completion_notified"):
						event = f"all-done-{time.time_ns()}"
						message = (
							f"Neu-sbox watchdog: all tracked tasks for {user} on {', '.join(hosts)} are terminal. "
							f"Read {state_path.resolve()} and task-logs for outcomes; done does not imply success. "
							"Continue the active experiment plan; do not resubmit completed tasks."
						)
						receipt = notify(output / event, thread, message, "all_tasks_done")
						state["completion_notified"] = receipt.get("queue_exit") == 0
						if not state["completion_notified"]:
							raise RuntimeError("Completion notification rejected; see callback receipt")
					state["status"] = "all_done"
					return
				if not owner.wait(min(interval, max(0, deadline - time.monotonic()))):
					break
			state["status"] = "owner_gone" if not owner.alive() else ("stopped" if stop.is_set() else "window_elapsed")
		finally:
			for signum, handler in previous_handlers.items():
				signal.signal(signum, handler)
			state["exited_unix"] = time.time()
			_atomic_write(state_path, state)


def main(
	output: Annotated[Path, typer.Option()],
	host: Annotated[list[str] | None, typer.Option("--host")] = None,
	user: str = "lhr",
	thread: str | None = None,
	seconds: Annotated[float, typer.Option(min=0.01, max=3600)] = 3600,
	interval: Annotated[float, typer.Option(min=0.01, max=300)] = 60,
):
	"""Bounded monitor; reuse output to resume. '*' selects all users explicitly."""
	thread = thread or os.environ.get("CODEX_THREAD_ID")
	if not thread:
		raise typer.BadParameter("Provide --thread or CODEX_THREAD_ID")
	monitor(output, thread, host or ["local"], user, seconds, interval, SessionOwner.discover(thread))


if __name__ == "__main__":
	typer.run(main)
