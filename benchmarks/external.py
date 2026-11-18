"""Shared runner and capability contract for independent external baselines."""

import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from benchmarks.sources import PinnedSource


@dataclass(frozen=True)
class ExternalArtifact:
	name: str
	validator_module: str | None
	capabilities: dict[str, Any]


_ARTIFACTS = {
	"ripple": ExternalArtifact(
		"ripple",
		"scripts.validate.ripple",
		{
			"artifact": "Ripple",
			"published": "ICDCS 2025; RIPPLE++ arXiv extension",
			"exact_operator": "author CustomGraphConvMean GC-mean",
			"exact_math": "author GC-mean; not NeutronRT GraphSAGE mean",
			"author_supported": {"updates": ["insert", "delete", "feature", "noop", "zero-indegree"]},
			"adapter_supported": {"updates": ["insert", "delete", "feature", "noop"]},
			"numerically_verified": {"updates": ["insert", "delete", "feature", "noop", "zero-indegree"]},
			"graph_restrictions": ["generated directed simple graph", "no self-loops in the gate"],
			"execution": {"cpu": True, "gpu": False, "gpu_comparability_validated": False},
			"environment_patches": [],
			"diagnostic_patch": None,
			"adaptation_scope": ["redirect generated model/state paths into the new output directory"],
			"numerical_verification": True,
			"stateful_replay": True,
			"performance_replay": True,
			"timing_available": True,
			"replay_gate": "gc_mean_zero_self_checkpoint_and_recorded_workload",
			"performance_readiness": "standalone_cpu_replay_ready_pending_matched_cohort",
			"limitations": [
				"The pinned commit is the author GC-mean implementation, not proof of every RIPPLE++ innovation.",
				"Replay accepts only the explicit GC-mean zero-self GraphSAGE compatibility checkpoint.",
			],
		},
	),
	"inkstream": ExternalArtifact(
		"inkstream",
		"scripts.validate.inkstream",
		{
			"artifact": "InkStream",
			"published": "IPDPS 2025",
			"exact_operator": "author SAGE with unchanged patched PyG SAGEConv",
			"exact_math": "author mean GraphSAGE path only",
			"author_supported": {"updates": ["insert", "delete", "feature"], "aggregations": ["mean", "sum", "max"]},
			"adapter_supported": {"updates": ["insert", "delete"], "aggregations": ["mean"]},
			"numerically_verified": {"updates": ["insert", "delete"], "aggregations": ["mean"]},
			"graph_restrictions": [
				"generated directed simple graph",
				"multiedges are rejected",
				"no self-loops in the gate",
			],
			"execution": {"cpu": True, "gpu": False, "gpu_comparability_validated": False},
			"environment_patches": ["authors' unchanged PyG 2.5.3 SAGE patch in isolated dependency environment"],
			"diagnostic_patch": None,
			"adaptation_scope": ["generated adjacency/intermediate fixtures", "observation of discarded outputs"],
			"numerical_verification": True,
			"stateful_replay": False,
			"performance_replay": False,
			"timing_available": False,
			"performance_readiness": "validation_failed_verification_only",
			"limitations": [
				"The unchanged artifact gate currently fails at the final-output boundary.",
				"The existing observer resets each batch; it is not a successive-state gate.",
				"No diagnostic patch is applied by this runner.",
			],
		},
	),
	"helios": ExternalArtifact(
		"helios",
		None,
		{
			"artifact": "Helios / Graph-Learn Dynamic Graph Service",
			"published": "PPoPP 2025",
			"source_path": "dynamic_graph_service",
			"author_supported": {"mechanism_evidence": "Graph-Learn source declares timestamp-top-k sampling"},
			"adapter_supported": {},
			"numerically_verified": {},
			"execution": {"cpu": False, "gpu": False, "gpu_comparability_validated": False},
			"environment_patches": [],
			"diagnostic_patch": None,
			"numerical_verification": False,
			"stateful_replay": False,
			"performance_replay": False,
			"timing_available": False,
			"performance_readiness": "unsupported_sampling_law_and_unverified_system_build",
			"performance_blocker": (
				"The pinned paper-linked source exposes timestamp-top-k, not a verified sampler matching this "
				"workload; no compatible Helios runner or full-system build has passed admission."
			),
			"limitations": [
				"A source pin is not evidence that the paper's complete Helios system was reproduced.",
				"Do not compare timestamp-top-k with uniform neighbor sampling or query-only latency.",
			],
		},
	),
}


def artifact(name: str) -> ExternalArtifact:
	try:
		return _ARTIFACTS[name]
	except KeyError as error:
		raise ValueError(f"Unknown external artifact: {name}") from error


def capability_manifest(name: str) -> dict[str, Any]:
	"""Return a detached capability manifest suitable for JSON reports."""
	return copy.deepcopy(artifact(name).capabilities)


def _git(checkout: Path, *arguments: str, timeout: float) -> bytes:
	result = subprocess.run(
		["git", "-C", str(checkout), *arguments],
		check=True,
		capture_output=True,
		timeout=max(0.01, timeout),
	)
	return result.stdout


def source_identity(name: str, checkout: Path, *, timeout: float = 30.0) -> dict[str, Any]:
	"""Validate the exact pinned checkout without fetching or modifying it."""
	pin = PinnedSource.load(name)
	checkout = checkout.resolve()
	if not checkout.is_dir():
		raise ValueError(f"External source checkout does not exist: {checkout}")
	deadline = monotonic() + timeout
	revision = _git(checkout, "rev-parse", "HEAD", timeout=deadline - monotonic()).decode().strip()
	if revision != pin.revision:
		raise ValueError(f"{name} requires upstream revision {pin.revision}, found {revision}")
	if _git(checkout, "status", "--porcelain", "--untracked-files=all", timeout=deadline - monotonic()):
		raise ValueError(f"{name} checkout has local changes; use an unchanged pinned checkout")
	# Hash the archive through a temporary file so a large checkout is not copied into
	# the Python heap. The subprocess is still bounded by the caller's deadline.
	with tempfile.TemporaryFile() as archive_file:
		subprocess.run(
			["git", "-C", str(checkout), "archive", "--format=tar", pin.revision],
			check=True,
			stdout=archive_file,
			stderr=subprocess.PIPE,
			timeout=max(0.01, deadline - monotonic()),
		)
		archive_file.seek(0)
		digest = hashlib.sha256()
		for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
			if monotonic() > deadline:
				raise subprocess.TimeoutExpired("git archive hash", timeout)
			digest.update(chunk)
	return {
		"name": name,
		"repository": pin.repository,
		"revision": pin.revision,
		"checkout": str(checkout),
		"git_archive_sha256": digest.hexdigest(),
	}


def _hardware_identity(value: str | None) -> dict[str, Any] | None:
	if value is None:
		return None
	path = Path(value).expanduser()
	if path.is_file():
		return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
	return {"label": value}


def _terminate_group(process: subprocess.Popen, grace: float = 5.0) -> None:
	grace = max(0.01, grace)
	# The child is started in its own process group. Kill that owned group even if
	# the leader has already exited: descendants can otherwise retain resources.
	try:
		os.killpg(process.pid, signal.SIGTERM)
	except (OSError, ProcessLookupError):
		try:
			process.terminate()
		except (OSError, ProcessLookupError):
			return
	try:
		process.wait(timeout=grace)
	except subprocess.TimeoutExpired:
		try:
			os.killpg(process.pid, signal.SIGKILL)
		except (OSError, ProcessLookupError):
			try:
				process.kill()
			except (OSError, ProcessLookupError):
				return
		try:
			process.wait(timeout=grace)
		except subprocess.TimeoutExpired:
			return


def _classify_failure(returncode: int | None, stderr: str, verification: dict[str, Any] | None) -> str:
	if verification is not None and verification.get("valid") is False:
		return "validation_failed"
	if returncode == 0 and verification is not None and verification.get("valid") is True:
		return "passed"
	text = stderr.lower()
	if any(
		token in text
		for token in (
			"unsupported",
			"not available",
			"no module named",
			"module not found",
			"missing dependency",
			"use pyg",
			"requires torch_geometric",
			"cuda device is not available",
			"no cuda devices are available",
		)
	):
		return "unsupported"
	return "failed"


def _resolve_python(value: Path | str | None) -> tuple[Path, str]:
	requested = str(value) if value is not None else sys.executable
	path = Path(requested).expanduser()
	if value is None:
		resolved = Path(sys.executable).resolve()
	elif path.is_absolute() or "/" in requested or "\\" in requested:
		if not path.is_file() or not os.access(path, os.X_OK):
			raise FileNotFoundError(f"Requested Python executable is missing or not executable: {path}")
		resolved = path.resolve()
	else:
		found = shutil.which(requested)
		if found is None:
			raise FileNotFoundError(f"Python executable was not found on PATH: {requested}")
		resolved = Path(found).resolve()
	digest = hashlib.sha256()
	with resolved.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return resolved, digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
	if not path.is_file():
		return None
	try:
		value = json.loads(path.read_text())
	except (OSError, json.JSONDecodeError):
		return None
	return value if isinstance(value, dict) else None


def _write_exclusive(path: Path, value: str | dict[str, Any]) -> None:
	with path.open("x", encoding="utf-8") as stream:
		if isinstance(value, dict):
			json.dump(value, stream, indent=2)
			stream.write("\n")
		else:
			stream.write(value)


def run_external(
	name: str,
	checkout: Path,
	output: Path,
	*,
	python_executable: Path | None = None,
	device: str = "cpu",
	hardware_profile: str | None = None,
	timeout: float = 300.0,
	seed: int = 71,
	mode: str = "verify",
	replay_options: dict[str, str | Path | bool | None] | None = None,
) -> dict[str, Any]:
	"""Run one external numerical gate in a fresh, bounded subprocess."""
	entry = artifact(name)
	output = output.resolve()
	if output.exists():
		raise ValueError(f"Output directory already exists; refusing to overwrite: {output}")
	if timeout <= 0 or timeout > 3600:
		raise ValueError("timeout must be greater than zero and at most 3600 seconds")
	if device != "cpu" and not device.startswith("cuda:"):
		raise ValueError("device must be cpu or an explicit cuda index")
	output.parent.mkdir(parents=True, exist_ok=True)
	stage = output.parent / f".{output.name}.external-runner"
	if stage.exists():
		raise ValueError(f"Runner staging directory already exists; refusing to overwrite: {stage}")
	stage.mkdir()
	base: dict[str, Any] = {
		"schema_version": 1,
		"artifact": name,
		"mode": mode,
		"status": "failed",
		"output": str(output),
		"python_executable_requested": str(python_executable) if python_executable is not None else None,
		"device": device,
		"hardware_profile": _hardware_identity(hardware_profile),
		"hardware_profile_enforced": False,
		"hardware_profile_note": "metadata label only; allocation is enforced by the parent Neu-sbox job",
		"timeout_seconds": timeout,
		"seed": seed,
		"capabilities": capability_manifest(name),
	}
	verification = None
	returncode: int | None = None
	stderr = ""
	try:
		deadline = monotonic() + timeout
		try:
			resolved_python, python_hash = _resolve_python(python_executable)
			base["python_executable"] = str(resolved_python)
			base["python_executable_sha256"] = python_hash
		except Exception as error:
			base.update(status="failed", failure=f"{type(error).__name__}: {error}", failure_phase="python_preflight")
			return base
		try:
			base["source"] = source_identity(name, checkout, timeout=max(0.01, deadline - monotonic()))
		except subprocess.TimeoutExpired as error:
			base.update(
				status="timeout",
				returncode=124,
				failure=f"Git source preflight timed out: {error}",
				failure_phase="source_preflight",
			)
			return base
		except Exception as error:
			base.update(status="failed", failure=f"{type(error).__name__}: {error}", failure_phase="source_preflight")
			return base
		if mode not in ("verify", "replay"):
			base.update(
				status="unsupported",
				failure=(
					"Supported modes are verify and the gated Ripple replay; no generic performance fallback exists."
				),
			)
			return base
		if mode == "replay" and not entry.capabilities.get("performance_replay", False):
			base.update(
				status="unsupported",
				failure=entry.capabilities.get(
					"performance_blocker", f"{name} has no implemented performance replay gate"
				),
				failure_phase="compatibility_gate",
			)
			return base
		if entry.validator_module is None:
			base.update(
				status="unsupported",
				failure=entry.capabilities.get("performance_blocker", f"{name} has no admitted verification adapter"),
				failure_phase="compatibility_gate",
			)
			return base
		if device != "cpu" and not entry.capabilities["execution"].get("gpu", False):
			base.update(
				status="unsupported",
				failure=f"{name} has no validated GPU adapter; CPU verification is the only supported execution mode.",
			)
			return base
		env = os.environ.copy()
		env["NEUTRONRT_EXTERNAL_DEVICE"] = device
		env["NEUTRONRT_EXTERNAL_HARDWARE_PROFILE"] = hardware_profile or ""
		if device == "cpu":
			env["CUDA_VISIBLE_DEVICES"] = ""
		elif "CUDA_VISIBLE_DEVICES" not in env:
			env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
		command = [
			str(resolved_python),
			"-m",
			"benchmarks.external_ripple_replay" if mode == "replay" else entry.validator_module,
			"--source",
			str(checkout.resolve()),
			"--output",
			str(output),
			"--seed",
			str(seed),
		]
		if mode == "replay":
			for key, value in (replay_options or {}).items():
				if isinstance(value, bool):
					if value:
						command.append(f"--{key.replace('_', '-')}")
				elif value is not None:
					command.extend([f"--{key.replace('_', '-')}", str(value)])
		base["command"] = command
		with (
			(stage / "stdout.log").open("w", encoding="utf-8") as out,
			(stage / "stderr.log").open("w", encoding="utf-8") as err,
		):
			try:
				process = subprocess.Popen(
					command,
					cwd=Path(__file__).resolve().parents[1],
					env=env,
					stdout=out,
					stderr=err,
					start_new_session=True,
				)
				try:
					remaining = max(0.01, deadline - monotonic())
					process.wait(timeout=remaining)
				except subprocess.TimeoutExpired:
					_terminate_group(process, grace=min(5.0, max(0.01, deadline - monotonic())))
					returncode = 124
				else:
					returncode = process.returncode
					_terminate_group(process, grace=min(1.0, max(0.01, deadline - monotonic())))
			except OSError as error:
				base["failure"] = f"{type(error).__name__}: {error}"
				returncode = None
		stderr = (stage / "stderr.log").read_text(encoding="utf-8")
		verification = _read_json(output / "verification.json")
		if returncode == 124:
			base["status"] = "timeout"
		else:
			base["status"] = _classify_failure(returncode, stderr, verification)
		if verification is not None:
			base["verification"] = {
				"valid": verification.get("valid"),
				"scope": verification.get("scope"),
				"dependencies": verification.get("dependencies"),
				"implementation_hashes": verification.get("implementation_hashes"),
			}
			for key in ("operator", "replay", "artifact_gate", "diagnostic_patch"):
				if key in verification:
					base["verification"][key] = verification[key]
			boundaries = sorted(
				{
					str(record["failure_boundary"])
					for record in (verification.get("records") or [])
					if isinstance(record, dict) and str(record["failure_boundary"])
				}
			)
			if boundaries:
				base["failure_boundaries"] = boundaries
		if base["status"] != "passed":
			base.setdefault("failure", stderr[-4000:] or f"external validator exited {returncode}")
		base["returncode"] = returncode
		base["workflow_elapsed_seconds"] = timeout - max(0.0, deadline - monotonic())
		base["dependency_identity"] = verification.get("dependencies") if verification else None
		return base
	finally:
		if not output.exists():
			output.mkdir(parents=True)
		for source, name_in_output in (
			(stage / "stdout.log", "external.stdout.log"),
			(stage / "stderr.log", "external.stderr.log"),
		):
			destination = output / name_in_output
			if source.exists() and not destination.exists():
				source.replace(destination)
		if "returncode" not in base:
			base["returncode"] = returncode
		if "dependency_identity" not in base:
			base["dependency_identity"] = verification.get("dependencies") if verification else None
		_write_exclusive(output / "external-run.json", base)
		stage.rmdir()
