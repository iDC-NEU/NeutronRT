"""A bounded hourly notification timer, independent of task-monitor health."""

import fcntl
import os
import time
from pathlib import Path
from typing import Annotated

import typer

from scripts.neubox.notify import notify
from scripts.neubox.session import SessionOwner
from scripts.neubox.watch import _atomic_write


def run(output: Path, thread: str, interval: float, state: Path, owner: SessionOwner) -> dict:
	output.mkdir(parents=True, exist_ok=False)
	with (output / "timer.lock").open("a") as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		started = time.time()
		_atomic_write(
			output / "timer.json",
			{"started_unix": started, "due_unix": started + interval, "thread": thread, "owner": owner.record()},
		)
		if not owner.wait(interval):
			_atomic_write(output / "stopped.json", {"reason": "owner_gone", "unix": time.time()})
			return {"owner_gone": True}
		message = (
			f"Hourly neu-sbox timer: inspect {state.resolve()} and watchdog task-logs, "
			"verify monitor/process health and advance completed work. "
			"Do not manually poll tasks repeatedly. Renew one bounded watchdog if needed and "
			"start the next 3600-second timer so hourly notifications continue."
		)
		return notify(output / "notification", thread, message, "hourly_health_check")


def main(
	output: Annotated[Path, typer.Option()],
	state: Annotated[Path, typer.Option(help="Watchdog state to inspect on notification")],
	thread: str | None = None,
	interval: Annotated[float, typer.Option(min=0.01, max=3600)] = 3600,
):
	thread = thread or os.environ.get("CODEX_THREAD_ID")
	if not thread:
		raise typer.BadParameter("Provide --thread or CODEX_THREAD_ID")
	receipt = run(output, thread, interval, state, SessionOwner.discover(thread))
	if not receipt.get("owner_gone") and receipt.get("queue_exit") != 0:
		raise typer.Exit(1)


if __name__ == "__main__":
	typer.run(main)
