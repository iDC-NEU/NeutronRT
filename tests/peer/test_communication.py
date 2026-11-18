import json

import pytest
from typer.testing import CliRunner

from scripts.peer import communication
from scripts.peer import scheduler as scheduler_module
from scripts.peer.communication import (
	RESULT_MARKER,
	_parse_result,
	_quantile,
	_read_perftest_report,
	_resource_request,
	_run_rdma,
	_run_torch_pair,
	_scheduler_worker_command,
	_worker_command,
)
from scripts.peer.scheduler import Allocation, NeuSbox, ResourceRequest, SchedulerError


def test_quantile_interpolates_samples():
	assert _quantile([1.0, 2.0, 4.0, 8.0], 0.5) == 3.0
	assert _quantile([1.0, 2.0, 4.0, 8.0], 0.95) == pytest.approx(7.4)


def test_parse_result_uses_the_rank_zero_marker():
	assert _parse_result(f'warning\n{RESULT_MARKER}{{"ok": true}}\n') == {"ok": True}


def test_worker_command_repeats_payload_sizes():
	command = _worker_command(
		"python",
		"gloo-cpu",
		0,
		"10.66.0.4",
		29600,
		"ibs10",
		0,
		(1024, 4096),
		5,
		10,
		30,
	)
	assert command.count("--size") == 2
	assert command[command.index("--mode") + 1] == "gloo-cpu"


def test_perftest_report_accepts_trailing_comma(tmp_path):
	report = tmp_path / "perftest.json"
	report.write_text('{"results": {"bw_avg": 18.7,}}\n')

	parsed, note = _read_perftest_report(report)

	assert parsed["results"]["bw_avg"] == 18.7
	assert note is not None


def test_resource_requests_use_bounded_cpu_or_single_gpu_profiles():
	assert _resource_request("gloo-cpu") == ResourceRequest(cpu=4, memory_gib=8, gpus=0)
	assert _resource_request("rdma-host") == ResourceRequest(cpu=4, memory_gib=8, gpus=0)
	assert _resource_request("nccl-gpu") == ResourceRequest(cpu=4, memory_gib=16, gpus=1)
	assert _resource_request("rdma-gpu") == ResourceRequest(cpu=4, memory_gib=16, gpus=1)
	assert "scripts.peer.gpu_exec" in _scheduler_worker_command(["python", "worker"], gpu=True, python="graph-python")


class _FakeScheduler:
	def __init__(self, *, fail_running=False, fail_completed=False, result_failure=False):
		self.fail_running = fail_running
		self.fail_completed = fail_completed
		self.result_failure = result_failure
		self.submissions = []
		self.waits = []
		self.cleaned = []

	def submit(self, host, request, command, **kwargs):
		allocation = Allocation(host, f"task-{len(self.submissions)}", request, command)
		self.submissions.append((allocation, kwargs))
		return allocation

	def wait_running(self, allocations, max_seconds):
		self.waits.append(("running", [item.task_id for item in allocations], max_seconds))
		if self.fail_running:
			raise TimeoutError("mock monitor timeout")
		for allocation in allocations:
			allocation.state = "running"

	def wait_completed(self, allocations, max_seconds):
		self.waits.append(("completed", [item.task_id for item in allocations], max_seconds))
		for allocation in allocations:
			allocation.state = "failed" if self.fail_completed else "completed"
		if self.fail_completed:
			raise RuntimeError("mock task failed")

	def result(self, allocation):
		if self.result_failure:
			return "worker exited", {
				"state": "completed",
				"result": {"returncode": 1, "timed_out": False, "error": None},
				"log": "worker exited",
			}
		if allocation.task_id == "task-1":
			return 'NEUTRONRT_PEER_RESULT={"ok":true}', {
				"state": "completed",
				"result": {"returncode": 0, "timed_out": False, "error": None},
				"log": 'NEUTRONRT_PEER_RESULT={"ok":true}',
			}
		return "peer completed", {
			"state": "completed",
			"result": {"returncode": 0, "timed_out": False, "error": None},
			"log": "peer completed",
		}

	def cleanup(self, allocations):
		self.cleaned.extend(item.task_id for item in allocations)
		return []


def test_torch_pair_submits_peer_before_local_and_monitors_both_tasks(tmp_path):
	scheduler = _FakeScheduler()
	result = _run_torch_pair(
		tmp_path,
		"graph-python",
		"gpunode2",
		tmp_path / "peer",
		"peer-python",
		"gloo-cpu",
		"10.0.0.1",
		29600,
		"ibs10",
		"ibs10",
		0,
		0,
		(16,),
		0,
		1,
		5,
		tmp_path / "logs",
		scheduler=scheduler,
		monitor_window=10,
	)
	assert result["task_ids"] == {"local": "task-1", "peer": "task-0"}
	assert [allocation.host for allocation, _ in scheduler.submissions] == ["gpunode2", None]
	assert all(allocation.request == ResourceRequest(4, 8, 0) for allocation, _ in scheduler.submissions)
	assert scheduler.waits[0][1] == ["task-0"]
	assert scheduler.waits[-1][1] == ["task-0", "task-1"]


def test_torch_pair_gpu_uses_one_gpu_on_both_endpoints_and_visible_cuda_zero(tmp_path):
	scheduler = _FakeScheduler()
	_run_torch_pair(
		tmp_path,
		"graph-python",
		"gpunode2",
		tmp_path / "peer",
		"peer-python",
		"nccl-gpu",
		"10.0.0.1",
		29600,
		"ibs10",
		"ibs10",
		0,
		0,
		(16,),
		0,
		1,
		5,
		tmp_path / "logs",
		scheduler=scheduler,
	)
	assert all(allocation.request == ResourceRequest(4, 16, 1) for allocation, _ in scheduler.submissions)
	assert all("--device 0" in allocation.command for allocation, _ in scheduler.submissions)
	assert all("scripts.peer.gpu_exec" in allocation.command for allocation, _ in scheduler.submissions)


def test_torch_pair_reuses_a_validated_inherited_allocation_without_nested_submit(monkeypatch, tmp_path):
	scheduler = _FakeScheduler()
	completed = type(
		"Completed",
		(),
		{"returncode": 0, "stdout": 'NEUTRONRT_PEER_RESULT={"ok":true}', "stderr": ""},
	)
	local_calls = []

	def run_local(command, **kwargs):
		local_calls.append((command, kwargs))
		return completed()

	monkeypatch.setattr(communication.subprocess, "run", run_local)
	result = _run_torch_pair(
		tmp_path,
		"graph-python",
		"gpunode2",
		tmp_path / "peer",
		"peer-python",
		"gloo-cpu",
		"10.0.0.1",
		29600,
		"ibs10",
		"ibs10",
		0,
		0,
		(16,),
		0,
		1,
		5,
		tmp_path / "logs",
		scheduler=scheduler,
		inherited_allocation={"name": "sbx_lhr_test.slice"},
	)
	assert [allocation.host for allocation, _ in scheduler.submissions] == ["gpunode2"]
	assert result["task_ids"]["local"] == "inherited:sbx_lhr_test.slice"
	assert len(local_calls) == 1 and local_calls[0][1]["timeout"] == 5


def test_torch_pair_timeout_cleans_only_owned_tasks(tmp_path):
	scheduler = _FakeScheduler(fail_running=True)
	with pytest.raises(RuntimeError, match="mock monitor timeout"):
		_run_torch_pair(
			tmp_path,
			"graph-python",
			"gpunode2",
			tmp_path / "peer",
			"peer-python",
			"gloo-cpu",
			"10.0.0.1",
			29600,
			"ibs10",
			"ibs10",
			0,
			0,
			(16,),
			0,
			1,
			5,
			tmp_path / "logs",
			scheduler=scheduler,
			monitor_window=1,
		)
	assert scheduler.cleaned == ["task-0"]


def test_torch_pair_failure_cleans_both_owned_tasks(tmp_path):
	scheduler = _FakeScheduler(fail_completed=True)
	with pytest.raises(RuntimeError, match="mock task failed"):
		_run_torch_pair(
			tmp_path,
			"graph-python",
			"gpunode2",
			tmp_path / "peer",
			"peer-python",
			"gloo-cpu",
			"10.0.0.1",
			29600,
			"ibs10",
			"ibs10",
			0,
			0,
			(16,),
			0,
			1,
			5,
			tmp_path / "logs",
			scheduler=scheduler,
		)
	assert scheduler.cleaned == ["task-0", "task-1"]


def test_rdma_host_submits_two_cpu_workers_and_waits_for_completion(tmp_path):
	scheduler = _FakeScheduler()
	result = _run_rdma(
		"gpunode2",
		"rdma-host",
		"mlx5_0",
		0,
		(128,),
		1,
		5,
		tmp_path / "logs",
		38515,
		39515,
		peer_addr="10.0.0.2",
		local_root=tmp_path,
		peer_root=tmp_path / "peer",
		scheduler=scheduler,
	)
	assert len(result["rows"]) == 2
	assert all(row["task_ids"]["peer"].startswith("task-") for row in result["rows"])
	assert len(scheduler.submissions) == 4
	assert all(item.request == ResourceRequest(4, 8, 0) for item, _ in scheduler.submissions)
	assert all("scripts.peer.gpu_exec" not in item.command for item, _ in scheduler.submissions)


def test_rdma_gpu_uses_the_single_scheduler_visible_gpu_on_both_ends(tmp_path):
	scheduler = _FakeScheduler()
	_run_rdma(
		"gpunode2",
		"rdma-gpu",
		"mlx5_0",
		0,
		(128,),
		1,
		5,
		tmp_path / "logs",
		38515,
		39515,
		peer_addr="10.0.0.2",
		local_root=tmp_path,
		peer_root=tmp_path / "peer",
		scheduler=scheduler,
	)
	assert len(scheduler.submissions) == 4
	assert all(item.request == ResourceRequest(4, 16, 1) for item, _ in scheduler.submissions)
	assert all("scripts.peer.gpu_exec" in item.command for item, _ in scheduler.submissions)
	assert all("--use_cuda=0" in item.command for item, _ in scheduler.submissions)


def test_neusbox_monitor_tracks_queue_running_completed_and_exact_task_id(tmp_path):
	statuses = {"known": iter(("queued", "running", "completed", "completed"))}
	calls = []

	def run(command, **kwargs):
		calls.append((command, kwargs))
		if "submit" in command:
			return type("Completed", (), {"returncode": 0, "stdout": '{"task_id":"known"}', "stderr": ""})()
		state = next(statuses["known"])
		payload = {
			"state": state,
			"result": {"returncode": 0, "timed_out": False, "error": None},
			"log": "worker output",
		}
		return type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()

	scheduler = NeuSbox(run=run, sleep=lambda _: None, poll_interval=0.01)
	allocation = scheduler.submit(
		None,
		ResourceRequest(4, 8, 0),
		"echo worker",
		cwd=tmp_path,
		wall_seconds=10,
		ledger=tmp_path / "task.json",
		label="test",
	)
	submit_command, _ = calls[0]
	assert submit_command[submit_command.index("--device-num") + 1] == "0"
	scheduler.wait_running([allocation], 10)
	scheduler.wait_completed([allocation], 10)
	assert [call[call.index("result") + 1] for call, _ in calls if "result" in call] == ["known"] * 3
	assert all("status" not in call and "cancel" not in call for call, _ in calls)
	assert all(0 < kwargs["timeout"] <= 15 for _, kwargs in calls)
	assert scheduler.result(allocation)[0] == "worker output"
	assert json.loads((tmp_path / "task.json").read_text())["task_id"] == "known"


@pytest.mark.parametrize(
	("metadata", "message"),
	[
		({"returncode": 7, "timed_out": False, "error": None}, "returned 7"),
		({"returncode": 124, "timed_out": True, "error": None}, "time limit"),
		({"returncode": 0, "timed_out": False, "error": "worker error"}, "execution error"),
	],
)
def test_neusbox_result_checks_return_code_and_worker_timeout(tmp_path, metadata, message):
	responses = iter(
		[
			'{"task_id":"bad"}',
			json.dumps({"state": "completed", "result": metadata, "log": "failed"}),
		]
	)

	def run(command, **kwargs):
		return type("Completed", (), {"returncode": 0, "stdout": next(responses), "stderr": ""})()

	scheduler = NeuSbox(run=run)
	allocation = scheduler.submit(
		None,
		ResourceRequest(4, 16, 1),
		"echo worker",
		cwd=tmp_path,
		wall_seconds=5,
		ledger=tmp_path / "bad.json",
		label="gpu-test",
	)
	with pytest.raises(SchedulerError, match=message):
		scheduler.result(allocation)


def test_neusbox_bounds_queued_wait_and_cleans_only_its_task(tmp_path):
	clock = [0.0]
	calls = []

	def run(command, **kwargs):
		calls.append((command, kwargs))
		if "submit" in command:
			return type("Completed", (), {"returncode": 0, "stdout": '{"task_id":"owned"}', "stderr": ""})()
		if "result" in command:
			payload = {"state": "queued"}
			return type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
		return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

	scheduler = NeuSbox(
		run=run,
		clock=lambda: clock[0],
		sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
		poll_interval=0.25,
		command_timeout=1,
	)
	allocation = scheduler.submit(
		None,
		ResourceRequest(4, 8, 0),
		"echo worker",
		cwd=tmp_path,
		wall_seconds=5,
		ledger=tmp_path / "queued.json",
		label="queued-test",
	)
	with pytest.raises(TimeoutError, match="become running"):
		scheduler.wait_running([allocation], 1)
	assert scheduler.cleanup([allocation]) == []
	assert allocation.state == "cancelled"
	assert any("DELETE" in call[2] and call[-1] == "owned" for call, _ in calls if call[0] == "python3")


def test_neusbox_uses_owned_worker_api_delete_for_cancellation(tmp_path):
	calls = []

	def run(command, **kwargs):
		calls.append((command, kwargs))
		if "submit" in " ".join(command):
			return type("Completed", (), {"returncode": 0, "stdout": '{"task_id":"ours"}', "stderr": ""})()
		return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

	scheduler = NeuSbox(run=run, command_timeout=10)
	allocation = scheduler.submit(
		"gpunode2",
		ResourceRequest(4, 8, 0),
		"echo worker",
		cwd=tmp_path,
		wall_seconds=5,
		ledger=tmp_path / "task.json",
		label="test",
	)
	foreign = Allocation("gpunode2", "users-task", ResourceRequest(4, 8, 0), "echo user job")
	with pytest.raises(SchedulerError, match="not submitted by this coordinator"):
		scheduler.cancel(foreign)
	assert len(calls) == 1
	scheduler.cancel(allocation)
	cancel_command, cancel_kwargs = calls[-1]
	assert cancel_command[0] == "ssh"
	assert "DELETE" in cancel_command[-1]
	assert "ours" in cancel_command[-1]
	assert "/healthz" in cancel_command[-1]
	assert "neu-sbox --json cancel" not in cancel_command[-1]
	assert cancel_kwargs["timeout"] == 10
	assert allocation.state == "cancelled"


def test_scheduler_cli_exposes_read_only_result_and_wait(monkeypatch):
	class ReadOnlyScheduler:
		def snapshot(self, task_id, *, host=None):
			return {"task_id": task_id, "host": host, "state": "running", "log": "partial"}

		def __init__(self, **kwargs):
			pass

		def wait_terminal(self, task_id, *, host=None, max_seconds=3600):
			return {"task_id": task_id, "host": host, "state": "completed", "log": "done"}

	monkeypatch.setattr(scheduler_module, "NeuSbox", ReadOnlyScheduler)
	app = scheduler_module.create_app()
	result = CliRunner().invoke(app, ["result", "task-42", "--host", "gpunode2"])
	assert result.exit_code == 0, result.output
	assert json.loads(result.output) == {
		"task_id": "task-42",
		"host": "gpunode2",
		"state": "running",
		"log": "partial",
	}
	waited = CliRunner().invoke(app, ["wait", "task-42", "--monitor-window", "1"])
	assert waited.exit_code == 0, waited.output
	assert json.loads(waited.output)["state"] == "completed"
