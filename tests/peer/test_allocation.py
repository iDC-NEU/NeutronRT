import json

import pytest
from typer.testing import CliRunner

from scripts.peer import allocation, gpu_exec
from scripts.peer.allocation import (
	AllocationMetadataError,
	gpu_environment,
	match_current_sandbox,
	parse_memory_gib,
	resolve_current_allocation,
	validate_allocation_capacity,
)

_STATUS = {"container": None, "pid": 42, "sandbox": "sbx_lhr_test.slice"}
_RECORD = {
	"name": "sbx_lhr_test.slice",
	"owner": "lhr",
	"cpu": 4,
	"mem": "16G",
	"devices": ["195:1"],
	"pids": [42],
}
_LIST = {"current_sandbox": None, "sandboxes": [_RECORD]}
_NVIDIA_XML = """<nvidia_smi_log>
  <gpu><minor_number>0</minor_number><uuid>GPU-zero</uuid></gpu>
  <gpu><minor_number>1</minor_number><uuid>GPU-one</uuid></gpu>
</nvidia_smi_log>"""


def _completed(stdout):
	return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


def test_resolver_matches_status_and_maps_granted_minor_to_stable_uuid():
	calls = []

	def run(command, **kwargs):
		calls.append((command, kwargs))
		if command[-1] == "status":
			return _completed(json.dumps(_STATUS))
		if command[-1] == "list":
			return _completed(json.dumps(_LIST))
		return _completed(_NVIDIA_XML)

	allocation, uuids = resolve_current_allocation(run=run, pid=42)
	assert allocation == _RECORD
	assert uuids == ["GPU-one"]
	assert calls[-1][0] == ["nvidia-smi", "-q", "-x"]
	assert all(kwargs["timeout"] == 10 for _, kwargs in calls)


def test_resolver_uses_cgroup_when_status_has_no_sandbox_name():
	allocation = match_current_sandbox(
		{"container": None, "pid": 2705, "sandbox": None},
		_LIST,
		pid=2705,
		cgroup_text="0::/sandbox_sbx_lhr_test.slice\n",
	)
	assert allocation == _RECORD


def test_gpu_environment_sets_uuid_and_allocation_capacity_is_checked():
	environment = gpu_environment(_RECORD, run=lambda command, **kwargs: _completed(_NVIDIA_XML), environ={})
	assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-one"
	assert parse_memory_gib("16") == 16
	validate_allocation_capacity(_RECORD, cpu=4, memory_gib=16, gpus=1)
	with pytest.raises(AllocationMetadataError, match="requires exactly 0"):
		validate_allocation_capacity(_RECORD, cpu=4, memory_gib=8, gpus=0)
	with pytest.raises(AllocationMetadataError, match="requires cpu>=8"):
		validate_allocation_capacity(_RECORD, cpu=8, memory_gib=16, gpus=1)


def test_resolver_fails_closed_when_allocated_device_has_no_uuid_mapping():
	def run(command, **kwargs):
		if command[-1] == "status":
			return _completed(json.dumps(_STATUS))
		if command[-1] == "list":
			return _completed(json.dumps(_LIST))
		return _completed("<nvidia_smi_log/>")

	with pytest.raises(AllocationMetadataError, match="did not map"):
		resolve_current_allocation(run=run, pid=42)


def test_gpu_worker_bootstrap_sets_uuid_before_exec(monkeypatch):
	seen = []
	monkeypatch.setattr(gpu_exec, "resolve_current_allocation", lambda: (_RECORD, ["GPU-one"]))
	monkeypatch.setattr(gpu_exec.sys, "argv", ["gpu_exec", "--", "python", "worker.py"])

	class Executed(Exception):
		pass

	def execvpe(program, command, environment):
		seen.append((program, command, environment))
		raise Executed

	monkeypatch.setattr(gpu_exec.os, "execvpe", execvpe)
	with pytest.raises(Executed):
		gpu_exec.main()
	assert seen[0][0] == "python"
	assert seen[0][1] == ["python", "worker.py"]
	assert seen[0][2]["CUDA_VISIBLE_DEVICES"] == "GPU-one"


def test_gpu_worker_bootstrap_rejects_multiple_allocated_gpus(monkeypatch):
	monkeypatch.setattr(gpu_exec, "resolve_current_allocation", lambda: (_RECORD, ["GPU-one", "GPU-two"]))
	monkeypatch.setattr(gpu_exec.sys, "argv", ["gpu_exec", "--", "python", "worker.py"])
	with pytest.raises(AllocationMetadataError, match="exactly one"):
		gpu_exec.main()


def test_allocation_cli_reports_current_record_and_uuid_mapping(monkeypatch):
	monkeypatch.setattr(
		allocation,
		"resolve_current_allocation",
		lambda: (_RECORD, ["GPU-one"]),
	)
	result = CliRunner().invoke(allocation.create_app(), [])
	assert result.exit_code == 0, result.output
	assert json.loads(result.output) == {"allocation": _RECORD, "gpu_uuids": ["GPU-one"]}


def test_gpu_exec_typer_preserves_delimiter_and_variadic_command(monkeypatch):
	seen = []
	monkeypatch.setattr(gpu_exec, "resolve_current_allocation", lambda: (_RECORD, ["GPU-one"]))
	monkeypatch.setattr(gpu_exec.os, "execvpe", lambda program, command, env: seen.append((program, command, env)))
	result = CliRunner().invoke(gpu_exec.create_app(), ["--", "python", "-V"])
	assert result.exit_code == 0, result.output
	assert seen[0][0] == "python" and seen[0][1] == ["python", "-V"]
	assert seen[0][2]["CUDA_VISIBLE_DEVICES"] == "GPU-one"
