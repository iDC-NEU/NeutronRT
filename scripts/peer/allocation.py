"""Resolve the current Neu Box allocation to stable accelerator UUIDs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Callable


class AllocationMetadataError(RuntimeError):
	"""Raised when the current worker allocation cannot be identified safely."""


def _cgroup_sandbox_name(text: str) -> str | None:
	match = re.search(r"(?:^|/)sandbox_(?P<name>[^/]+\.slice)(?:/|$)", text, re.MULTILINE)
	return None if match is None else match.group("name")


def match_current_sandbox(
	status: dict[str, Any],
	listing: dict[str, Any],
	*,
	pid: int | None = None,
	cgroup_text: str | None = None,
) -> dict[str, Any] | None:
	"""Match `neu-sbox status` and `list` to this process's allocation record."""
	pid = os.getpid() if pid is None else pid
	name = status.get("sandbox")
	if name is not None and not isinstance(name, str):
		raise AllocationMetadataError("neu-sbox status returned an unsupported sandbox value")
	if not name:
		name = _cgroup_sandbox_name(_read_cgroup() if cgroup_text is None else cgroup_text)
	if not name:
		status_pid = status.get("pid")
		if type(status_pid) is int:
			pid = status_pid

	sandboxes = listing.get("sandboxes")
	if not isinstance(sandboxes, list):
		raise AllocationMetadataError("neu-sbox list omitted the sandboxes array")
	if name:
		matches = [item for item in sandboxes if isinstance(item, dict) and item.get("name") == name]
	else:
		matches = [
			item
			for item in sandboxes
			if isinstance(item, dict)
			and isinstance(item.get("pids"), list)
			and any(type(item_pid) is int and item_pid == pid for item_pid in item["pids"])
		]
	if not matches:
		if name:
			raise AllocationMetadataError(f"neu-sbox status named sandbox {name!r}, but list did not contain it")
		return None
	if len(matches) != 1:
		raise AllocationMetadataError("Could not uniquely match this process to a Neu Box sandbox")
	return matches[0]


def _read_cgroup() -> str:
	try:
		with open("/proc/self/cgroup", encoding="utf-8") as stream:
			return stream.read()
	except OSError:
		return ""


def _json_command(
	run: Callable[..., subprocess.CompletedProcess[str]], command: list[str], *, timeout: float
) -> dict[str, Any]:
	try:
		completed = run(command, capture_output=True, text=True, check=False, timeout=timeout)
	except (OSError, subprocess.TimeoutExpired) as error:
		raise AllocationMetadataError(f"Could not inspect the current Neu Box allocation: {error}") from error
	if completed.returncode != 0:
		raise AllocationMetadataError(
			f"Allocation inspection command failed with {completed.returncode}: {completed.stderr.strip()}"
		)
	try:
		payload = json.loads(completed.stdout)
	except json.JSONDecodeError as error:
		raise AllocationMetadataError(
			f"Allocation inspection returned invalid JSON: {completed.stdout.strip()}"
		) from error
	if not isinstance(payload, dict):
		raise AllocationMetadataError("Allocation inspection returned a non-object JSON response")
	return payload


def _device_minor(device: str) -> int | None:
	value = device.strip()
	if value.startswith("GPU-"):
		return None
	if ":" in value:
		major, minor = value.rsplit(":", 1)
		if not major.isdecimal():
			raise AllocationMetadataError(f"Unsupported allocated device identifier: {device!r}")
	else:
		minor = value
	if not minor.isdecimal():
		raise AllocationMetadataError(f"Unsupported allocated device identifier: {device!r}")
	return int(minor)


def _gpu_uuids(
	devices: list[Any],
	*,
	run: Callable[..., subprocess.CompletedProcess[str]],
	timeout: float,
) -> list[str]:
	if not devices:
		return []
	if any(not isinstance(device, str) for device in devices):
		raise AllocationMetadataError("Neu Box allocation contains a non-string device identifier")
	try:
		completed = run(
			["nvidia-smi", "-q", "-x"],
			capture_output=True,
			text=True,
			check=False,
			timeout=timeout,
		)
	except (OSError, subprocess.TimeoutExpired) as error:
		raise AllocationMetadataError(f"Could not map allocated devices with nvidia-smi: {error}") from error
	if completed.returncode != 0:
		raise AllocationMetadataError(f"nvidia-smi failed with {completed.returncode}: {completed.stderr.strip()}")
	try:
		root = ET.fromstring(completed.stdout)
	except ET.ParseError as error:
		raise AllocationMetadataError("nvidia-smi returned invalid XML") from error
	rows: list[tuple[int, str]] = []
	for gpu in root.findall(".//gpu"):
		minor_text = gpu.findtext("minor_number")
		uuid = gpu.findtext("uuid")
		if minor_text and minor_text.isdecimal() and uuid and uuid.startswith("GPU-"):
			rows.append((int(minor_text), uuid.strip()))
	result = []
	for device in devices:
		if device.startswith("GPU-"):
			matches = [uuid for _, uuid in rows if uuid == device]
		else:
			minor = _device_minor(device)
			matches = [uuid for row_minor, uuid in rows if row_minor == minor]
		if len(matches) != 1:
			raise AllocationMetadataError(f"Allocated device {device!r} did not map to exactly one nvidia-smi GPU UUID")
		result.append(matches[0])
	if len(set(result)) != len(result):
		raise AllocationMetadataError("Neu Box assigned the same GPU UUID more than once")
	return result


def parse_memory_gib(value: Any) -> float:
	"""Convert Neu Box list memory fields such as `16G` to GiB units."""
	if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
		return float(value)
	if not isinstance(value, str):
		raise AllocationMetadataError(f"Unsupported sandbox memory limit: {value!r}")
	match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|K|KB|KIB|M|MB|MIB|G|GB|GIB|T|TB|TIB)?\s*", value, re.IGNORECASE)
	if match is None:
		raise AllocationMetadataError(f"Unsupported sandbox memory limit: {value!r}")
	amount = float(match.group(1))
	unit = (match.group(2) or "").upper()
	factor = {
		"": 1,
		"G": 1,
		"GB": 1,
		"GIB": 1,
		"T": 1024,
		"TB": 1024,
		"TIB": 1024,
		"M": 1 / 1024,
		"MB": 1 / 1024,
		"MIB": 1 / 1024,
		"K": 1 / 1024**2,
		"KB": 1 / 1024**2,
		"KIB": 1 / 1024**2,
		"B": 1 / 1024**3,
	}
	return amount * factor[unit]


def validate_allocation_capacity(
	allocation: dict[str, Any],
	*,
	cpu: int,
	memory_gib: int,
	gpus: int,
) -> None:
	"""Require an inherited sandbox to meet a requested peer resource profile."""
	actual_cpu = allocation.get("cpu")
	actual_memory = allocation.get("mem")
	devices = allocation.get("devices")
	if not isinstance(allocation.get("name"), str) or type(actual_cpu) is not int or not isinstance(devices, list):
		raise AllocationMetadataError("Matched sandbox omitted its name, CPU, memory, or devices limits")
	if actual_cpu < cpu or parse_memory_gib(actual_memory) < memory_gib:
		raise AllocationMetadataError(
			f"Inherited sandbox has cpu={actual_cpu}, mem={actual_memory}; requires cpu>={cpu}, mem>={memory_gib}G"
		)
	if len(devices) != gpus:
		raise AllocationMetadataError(f"Inherited sandbox has {len(devices)} GPUs; this run requires exactly {gpus}")


def resolve_current_allocation(
	*,
	run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
	timeout: float = 10.0,
	pid: int | None = None,
	cgroup_text: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
	"""Return the matching sandbox record and granted GPU UUIDs, without importing Torch.

	The matching contract is `status.sandbox` (or `/proc/self/cgroup`'s
	`sandbox_<name>.slice`) against `list.sandboxes[].name`. A PID match against
	`sandboxes[].pids` is the fallback when neither name is available. GPU IDs in
	`devices` are mapped by minor number to UUIDs from `nvidia-smi -q -x`.
	"""
	status = _json_command(run, ["neu-sbox", "--json", "status"], timeout=timeout)
	listing = _json_command(run, ["neu-sbox", "--json", "list"], timeout=timeout)
	allocation = match_current_sandbox(status, listing, pid=pid, cgroup_text=cgroup_text)
	if allocation is None:
		return None, []
	devices = allocation.get("devices")
	if not isinstance(devices, list):
		raise AllocationMetadataError("Matched sandbox record omitted its devices array")
	return allocation, _gpu_uuids(devices, run=run, timeout=timeout)


def gpu_environment(
	allocation: dict[str, Any],
	*,
	run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
	timeout: float = 10.0,
	environ: dict[str, str] | None = None,
) -> dict[str, str]:
	"""Build an environment exposing exactly the allocated GPUs by stable UUID."""
	devices = allocation.get("devices")
	if not isinstance(devices, list) or not devices:
		raise AllocationMetadataError("GPU worker requires a non-empty allocated devices list")
	uuids = _gpu_uuids(devices, run=run, timeout=timeout)
	result = dict(os.environ if environ is None else environ)
	result["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
	return result


def create_app():
	"""Build the optional Typer CLI without adding Typer to resolver imports."""
	import typer

	app = typer.Typer(no_args_is_help=True, help="Inspect the current Neu Box allocation and its GPU UUID mapping.")

	@app.command("current")
	def current(
		require_allocation: bool = typer.Option(False, help="Fail when this process has no allocation"),
	) -> None:
		try:
			allocation, gpu_uuids = resolve_current_allocation()
		except AllocationMetadataError as error:
			typer.echo(str(error), err=True)
			raise typer.Exit(1) from error
		if allocation is None and require_allocation:
			typer.echo("This process is not inside a Neu Box allocation", err=True)
			raise typer.Exit(1)
		typer.echo(json.dumps({"allocation": allocation, "gpu_uuids": gpu_uuids}, indent=2))

	return app


def cli() -> None:
	create_app()()


if __name__ == "__main__":
	cli()
