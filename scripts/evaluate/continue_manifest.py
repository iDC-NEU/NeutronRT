"""Bounded lightweight monitor that submits one dependent manifest exactly once."""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from scripts.evaluate.snapshot import digest


def save(path: Path, state: dict):
	temporary = path.with_suffix(".tmp")
	temporary.write_text(json.dumps(state, indent=2) + "\n")
	temporary.replace(path)


def poll(task_id: str) -> dict:
	result = subprocess.run(
		["neu-sbox", "--json", "result", task_id], capture_output=True, text=True, check=True, timeout=30
	)
	task = json.loads(result.stdout)
	return {key: task.get(key) for key in ("task_id", "status", "result", "started_at", "finished_at")}


def step(plan: dict, state: dict, path: Path) -> bool:
	if state.get("finished"):
		return True
	if not state.get("task_id"):
		for key, filename in (("manifest", plan["manifest"]), ("hardware", plan.get("hardware"))):
			if filename:
				actual = digest(Path(filename))
				field = key + "_sha256"
				if field in state and state[field] != actual:
					raise ValueError(f"Continuation {key} changed while waiting; no job submitted")
				state[field] = actual
		state["prerequisites"] = [poll(task) for task in plan["after_tasks"]]
		save(path, state)
		if any(task["status"] not in ("completed", "failed", "cancelled") for task in state["prerequisites"]):
			return False
		# A failed old wrapper may still have completed its transfer. Explicit
		# artifact requirements, not the old wrapper exit code, gate preparation.
		manifest = json.loads(Path(plan["manifest"]).read_text())
		for name in manifest.get("required_files", []):
			if not Path(name).is_file():
				raise FileNotFoundError(f"Prerequisite artifact is missing; no job submitted: {name}")
		result = subprocess.run(
			[
				sys.executable,
				"-m",
				"scripts.evaluate.run_manifest",
				plan["manifest"],
				"--output",
				plan["output"],
				*(["--hardware", plan["hardware"]] if plan.get("hardware") else []),
				"--submit",
			],
			capture_output=True,
			text=True,
			check=True,
			timeout=120,
		)
		state["task_id"] = json.loads(result.stdout)["task_id"]
		save(path, state)
		print(json.dumps({"submitted": state["task_id"]}), flush=True)
	state["task"] = poll(state["task_id"])
	state["finished"] = state["task"]["status"] in ("completed", "failed", "cancelled")
	save(path, state)
	return bool(state["finished"])


def main(
	plan: Annotated[Path, typer.Argument()],
	state: Annotated[Path, typer.Option("--state")],
	seconds: Annotated[float, typer.Option("--seconds", min=0.001, max=3600)] = 3600,
):
	settings = json.loads(plan.read_text())
	state.parent.mkdir(parents=True, exist_ok=True)
	with state.with_suffix(".lock").open("a") as lock:
		fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
		saved = json.loads(state.read_text()) if state.exists() else {"plan": settings}
		if saved["plan"] != settings:
			raise ValueError("Continuation plan differs from saved state")
		deadline = time.monotonic() + seconds
		while time.monotonic() < deadline:
			if step(settings, saved, state):
				print(json.dumps(saved["task"]), flush=True)
				raise SystemExit(0 if (saved["task"].get("result") or {}).get("returncode") == 0 else 1)
			time.sleep(min(30, max(0, deadline - time.monotonic())))
		print("Monitor window elapsed; check saved task health and resume this same state", flush=True)
		raise SystemExit(124)


if __name__ == "__main__":
	typer.run(main)
