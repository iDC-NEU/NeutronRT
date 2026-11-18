"""Run frozen trials serially in one allocation, retaining failures and timeouts."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer


def save(path: Path, value: dict):
	temporary = path.with_suffix(".tmp")
	temporary.write_text(json.dumps(value, indent=2) + "\n")
	temporary.replace(path)


def stop(process):
	# torchrun and storage/worker descendants must release the same allocation.
	for sig in (signal.SIGTERM, signal.SIGKILL):
		try:
			os.killpg(process.pid, sig)
		except ProcessLookupError:
			break
		if sig == signal.SIGTERM:
			time.sleep(2)
	process.wait(timeout=30)


def run_trial(command: list[str], output: Path, wall: float, startup: float) -> dict:
	output.mkdir(parents=True, exist_ok=False)
	phase_path = output / "phase.json"
	command = [sys.executable if part == "{python}" else part.replace("{trial}", str(output)) for part in command]
	started = time.monotonic()
	progress, ready, censor = {}, False, None
	with (output / "console.log").open("w") as log:
		process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		try:
			while process.poll() is None:
				if phase_path.exists():
					progress = json.loads(phase_path.read_text())
					ready |= progress.get("phase") not in ("setup", "loading", "unknown", None)
				elapsed = time.monotonic() - started
				if elapsed >= wall or (not ready and elapsed >= startup):
					censor = "wall" if elapsed >= wall else "startup"
					stop(process)
					break
				time.sleep(1)
		finally:
			# A failed launcher can exit before its descendants do.
			stop(process)
	if phase_path.exists():
		progress = json.loads(phase_path.read_text())
	status = {
		"command": command,
		"returncode": process.returncode,
		"timed_out": censor is not None,
		"timeout_kind": censor,
		"timeout_seconds": (wall if censor == "wall" else startup) if censor else None,
		"observed_seconds": time.monotonic() - started,
		"phase": progress.get("phase", "unknown"),
		"method": progress.get("method"),
		"repetition": progress.get("repetition"),
	}
	save(output / "process-status.json", status)
	return status


def main(plan: Annotated[Path, typer.Argument()], output: Annotated[Path, typer.Option("--output")]):
	settings = json.loads(plan.read_text())
	trials = settings["trials"]
	if not trials or len({trial["id"] for trial in trials}) != len(trials):
		raise ValueError("Trial IDs must be unique and nonempty")
	for trial in trials:
		if Path(trial["id"]).name != trial["id"] or trial["id"] in (".", ".."):
			raise ValueError("Trial IDs must be plain directory names")
	if not 0 < settings["startup_seconds"] <= settings["wall_seconds"]:
		raise ValueError("Invalid trial deadlines")
	output.mkdir(parents=True, exist_ok=False)
	save(output / "plan.json", settings)
	results = []
	for trial in trials:
		print(json.dumps({"starting": trial["id"]}), flush=True)
		status = run_trial(
			trial["command"], output / trial["id"], settings["wall_seconds"], settings["startup_seconds"]
		)
		results.append({"id": trial["id"], **status})
		save(output / "summary.json", {"trials": results})
		print(json.dumps({"finished": trial["id"], **status}), flush=True)
	raise SystemExit(0 if all(item["returncode"] == 0 and not item["timed_out"] for item in results) else 1)


if __name__ == "__main__":
	typer.run(main)
