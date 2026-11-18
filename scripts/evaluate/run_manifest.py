"""Execute a frozen command manifest within an existing scheduler allocation."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from scripts.evaluate.snapshot import digest, verify


@dataclass
class Options:
	manifest: Path
	output: Path
	submit: bool = False
	hardware: Path = Path("config/experiments/hardware.json")
	expected_manifest_sha256: str | None = None


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def execute(
	manifest: Annotated[Path, typer.Argument()],
	output: Annotated[Path, typer.Option("--output")],
	submit: Annotated[bool, typer.Option("--submit")] = False,
	hardware: Annotated[Path, typer.Option("--hardware")] = Path("config/experiments/hardware.json"),
	expected_manifest_sha256: Annotated[str | None, typer.Option("--expected-manifest-sha256")] = None,
):
	_run(Options(manifest, output, submit, hardware, expected_manifest_sha256))


def main():
	app(standalone_mode=False)


def _run(args: Options):
	if args.expected_manifest_sha256 and digest(args.manifest) != args.expected_manifest_sha256:
		raise ValueError("Frozen submission manifest changed")
	manifest = json.loads(args.manifest.read_text())
	for name in manifest.get("required_files", []):
		if not Path(name).is_file():
			raise FileNotFoundError(f"Required input is missing: {name}")
	if args.submit:
		from lib.profiling.submission import submit_once

		task = submit_once(args.manifest, args.hardware, args.output, sys.executable)
		print(json.dumps(task, indent=2))
		return
	if "source_snapshot" in manifest and verify(Path.cwd()) != manifest["source_snapshot"]:
		raise ValueError("Frozen source metadata changed")
	for name, expected in manifest.get("expected_inputs_sha256", {}).items():
		if digest(Path(name)) != expected:
			raise ValueError(f"Frozen input changed: {name}")
	command = [
		sys.executable if item == "{python}" else item.replace("{output}", str(args.output))
		for item in manifest["command"]
	]
	env = os.environ | manifest.get("environment", {})
	allocation = None
	if "hardware" in manifest:
		from scripts.peer.allocation import resolve_current_allocation

		row, gpu_uuids = resolve_current_allocation()
		profile = manifest["hardware"]
		if row is None:
			raise RuntimeError("Frozen jobs must execute in their neu-sbox allocation")
		if (int(row.get("cpu", 0)), row.get("mem"), len(gpu_uuids)) != (
			profile["cpu"],
			f"{profile['mem']}G",
			profile["gpus"],
		):
			raise RuntimeError(f"Actual allocation differs from frozen hardware request: {row}")
		env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
		allocation = {"sandbox": row, "gpu_uuids": gpu_uuids}

	args.output.mkdir(parents=True, exist_ok=False)

	def git(*parts):
		return subprocess.check_output(["git", *parts], text=True, timeout=30)

	metadata = {
		"manifest": manifest,
		"allocation": allocation,
		"command": command,
		"python": sys.executable,
		"host": platform.node(),
		"platform": platform.platform(),
		"commit": manifest["source_snapshot"]["commit"]
		if "source_snapshot" in manifest
		else git("rev-parse", "HEAD").strip(),
		"packages": {name: importlib.metadata.version(name) for name in ("torch", "dgl", "numpy", "psutil")},
		"started_unix": time.time(),
	}
	(args.output / "source.patch").write_bytes(
		Path("source.patch").read_bytes()
		if "source_snapshot" in manifest
		else subprocess.check_output(["git", "diff", "--binary", "HEAD"])
	)
	metadata["inputs_sha256"] = {}
	for value in manifest.get("hash_files", []):
		path = Path(value)
		with path.open("rb") as stream:
			metadata["inputs_sha256"][value] = hashlib.file_digest(stream, "sha256").hexdigest()
	for name, cmd in {"cpu": ["lscpu"], "memory": ["free", "-b"], "gpu": ["nvidia-smi", "-q"]}.items():
		try:
			result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
			text = result.stdout + result.stderr
		except (OSError, subprocess.TimeoutExpired) as error:
			text = f"Hardware metadata unavailable: {error}\n"
		(args.output / f"{name}.txt").write_text(text)
	(args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
	child_started = time.monotonic()
	with (args.output / "console.log").open("w") as log:
		result = subprocess.run(
			["timeout", "--foreground", "--kill-after=30s", str(manifest["wall_seconds"]), *command],
			env=env,
			text=True,
			stdout=log,
			stderr=subprocess.STDOUT,
		)
	child_seconds = time.monotonic() - child_started
	timed_out = result.returncode in (124, 137) and child_seconds >= manifest["wall_seconds"] - 0.1
	progress = {}
	if manifest.get("phase_file"):
		phase_path = Path(manifest["phase_file"].replace("{output}", str(args.output)))
		if phase_path.is_file():
			try:
				progress = json.loads(phase_path.read_text())
				if not isinstance(progress, dict):
					raise ValueError("Phase record must be an object")
			except (OSError, ValueError) as error:
				progress = {"phase": "unknown", "error": str(error)}
	(args.output / "process-status.json").write_text(
		json.dumps(
			{
				"returncode": result.returncode,
				"method": progress.get("method", manifest.get("method")),
				"phase": progress.get("phase", "unknown"),
				"phase_error": progress.get("error"),
				"cohort_identity": progress.get("cohort_identity", manifest.get("cohort_identity")),
				"manifest_sha256": digest(args.manifest),
				"source_tree_sha256": manifest.get("source_snapshot", {}).get("tree_sha256"),
				"timed_out": timed_out,
				"timeout_seconds": manifest["wall_seconds"] if timed_out else None,
				"observed_child_seconds": child_seconds,
				"finished_unix": time.time(),
			},
			indent=2,
		)
		+ "\n"
	)
	raise SystemExit(result.returncode)


if __name__ == "__main__":
	main()
