import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _fake_programs(tmp_path):
	worker = tmp_path / "fake-watch-worker.py"
	worker.write_text(
		"""
import json
import os
import sys
import time

mode = os.environ.get("FAKE_WATCH_MODE", "expiry")
print(json.dumps({"mode": mode}), flush=True)
if mode == "hang":
    time.sleep(60)
elif mode == "failure":
    raise SystemExit(7)
elif mode == "success":
    raise SystemExit(0)
raise SystemExit(124)
""".strip()
		+ "\n"
	)
	codex = tmp_path / "fake-codex.py"
	codex.write_text(
		"""
#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

receipt = Path(os.environ["NEUTRONRT_WATCH_RECEIPT"])
Path(os.environ["FAKE_CODEX_OBSERVED"]).write_text(receipt.read_text())
mode = os.environ.get("FAKE_CODEX_MODE", "success")
if mode == "failure":
    print("queue rejected", file=sys.stderr)
    raise SystemExit(9)
if mode == "timeout":
    time.sleep(2)
print("warning: active writer", file=sys.stderr)
""".strip()
		+ "\n"
	)
	codex.chmod(0o755)
	return worker, codex


def _command(output: Path, thread: str = "parent-thread", seconds: int = 1):
	return [
		sys.executable,
		"-m",
		"scripts.neubox.watch",
		"--task-id",
		"job-1",
		"--output",
		str(output),
		"--thread",
		thread,
		"--seconds",
		str(seconds),
	]


def _environment(worker: Path, codex: Path, observed: Path, *, worker_mode="expiry", codex_mode="success"):
	environment = os.environ.copy()
	environment.update(
		{
			"NEUTRONRT_WATCH_WORKER": str(worker),
			"NEUTRONRT_CODEX_EXECUTABLE": str(codex),
			"FAKE_WATCH_MODE": worker_mode,
			"FAKE_CODEX_MODE": codex_mode,
			"FAKE_CODEX_OBSERVED": str(observed),
		}
	)
	return environment


def _run(output, environment, *, timeout=8):
	return subprocess.run(
		_command(output), cwd=ROOT, env=environment, capture_output=True, text=True, timeout=timeout, check=False
	)


def test_real_subprocess_expiry_writes_receipt_before_single_callback(tmp_path):
	worker, codex = _fake_programs(tmp_path)
	output, observed = tmp_path / "expiry", tmp_path / "observed.json"
	result = _run(output, _environment(worker, codex, observed))
	assert result.returncode == 124, result.stderr
	receipt = json.loads((output / "receipt.json").read_text())
	before_callback = json.loads(observed.read_text())
	assert receipt["monitor_status"] == "worker_timeout"
	assert before_callback["monitor_exit"] == 124
	assert before_callback["callback_attempted"]
	assert receipt["callback_status"] == "queued"
	assert receipt["queue_warning"] and not receipt["resume_verified"]
	assert (output / "watch.jsonl").read_text()


def test_real_subprocess_worker_success_preserves_zero_exit_and_receipt(tmp_path):
	worker, codex = _fake_programs(tmp_path)
	output, observed = tmp_path / "success", tmp_path / "success-observed.json"
	result = _run(output, _environment(worker, codex, observed, worker_mode="success"))
	assert result.returncode == 0, result.stderr
	receipt = json.loads((output / "receipt.json").read_text())
	assert receipt["monitor_exit"] == receipt["wrapper_exit"] == 0
	assert receipt["monitor_status"] == "worker_completed"
	assert receipt["callback_status"] == "queued"
	assert json.loads(observed.read_text())["monitor_exit"] == 0


def test_real_subprocess_parent_sigterm_preserves_signal_and_cleans_owned_worker_group(tmp_path):
	worker, codex = _fake_programs(tmp_path)
	output, observed = tmp_path / "signal", tmp_path / "observed.json"
	process = subprocess.Popen(
		_command(output, seconds=5),
		cwd=ROOT,
		env=_environment(worker, codex, observed, worker_mode="hang"),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	time.sleep(0.3)
	process.send_signal(signal.SIGTERM)
	assert process.wait(timeout=5) == 143
	receipt = json.loads((output / "receipt.json").read_text())
	assert receipt["signal"] == signal.SIGTERM
	assert receipt["wrapper_exit"] == 143
	assert receipt["monitor_status"] == "interrupted"
	assert json.loads(observed.read_text())["monitor_exit"] in (-15, 143)


def test_real_subprocess_worker_failure_preserves_worker_exit_and_still_callbacks(tmp_path):
	worker, codex = _fake_programs(tmp_path)
	output, observed = tmp_path / "worker-failure", tmp_path / "observed.json"
	result = _run(output, _environment(worker, codex, observed, worker_mode="failure"))
	assert result.returncode == 7
	receipt = json.loads((output / "receipt.json").read_text())
	assert receipt["monitor_exit"] == 7 and receipt["monitor_status"] == "worker_failed"
	assert receipt["callback_status"] == "queued"


@pytest.mark.parametrize("codex_mode,expected_status", [("failure", "failed"), ("timeout", "timeout")])
def test_real_subprocess_queue_failure_or_timeout_does_not_replace_monitor_exit(
	tmp_path, monkeypatch, codex_mode, expected_status
):
	worker, codex = _fake_programs(tmp_path)
	output, observed = tmp_path / codex_mode, tmp_path / f"{codex_mode}-observed.json"
	if codex_mode == "timeout":
		monkeypatch.setenv("NEUTRONRT_WATCH_QUEUE_TIMEOUT", "0.1")
	result = _run(output, _environment(worker, codex, observed, codex_mode=codex_mode))
	assert result.returncode == 124
	receipt = json.loads((output / "receipt.json").read_text())
	assert receipt["monitor_exit"] == 124
	assert receipt["callback_status"] == expected_status
	assert not receipt["resume_verified"]
