"""Small finite-job coordinator for peer communication measurements."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

QUEUED_STATES = {"queued", "pending", "waiting", "submitted"}
RUNNING_STATES = {"running", "started", "executing"}
TERMINAL_STATES = {"completed", "failed", "cancelled", "canceled", "timeout", "timed_out", "error"}

_DELETE_TASK_SCRIPT = """\
import json, os, sys, urllib.request
base = os.environ.get('NEU_BOX_URL', 'http://127.0.0.1:59075').rstrip('/')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(base + '/healthz', timeout=5) as response:
    health = json.loads(response.read())
if health.get('role') != 'worker' or int(health.get('api_version', 0)) < 2:
    raise SystemExit('worker API v2 is required for owned-task cancellation')
url = base + '/tasks'
body = json.dumps({'task_ids': [sys.argv[1]]}, separators=(',', ':')).encode()
request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'}, method='DELETE')
with opener.open(request, timeout=5) as response:
    if not 200 <= response.status < 300:
        raise SystemExit('worker rejected owned-task cancellation: HTTP %s' % response.status)
    response.read()
"""


class SchedulerError(RuntimeError):
	"""A scheduler request or owned task failed."""


@dataclass(frozen=True)
class ResourceRequest:
	cpu: int
	memory_gib: int
	gpus: int

	def __post_init__(self) -> None:
		if self.cpu < 1 or self.memory_gib < 1 or self.gpus not in (0, 1):
			raise ValueError("Peer allocations require positive CPU/RAM and zero or one GPU")


@dataclass
class Allocation:
	host: str | None
	task_id: str
	request: ResourceRequest
	command: str
	state: str = "submitted"
	last_status: dict[str, Any] | None = None


def _state(payload: dict[str, Any]) -> str:
	value: Any = payload
	for _ in range(3):
		if isinstance(value, dict):
			for key in ("state", "status", "phase"):
				candidate = value.get(key)
				if isinstance(candidate, str):
					return candidate.lower()
			value = value.get("task")
		else:
			break
	return "unknown"


def _result_text(payload: dict[str, Any]) -> str:
	value = payload.get("log")
	if isinstance(value, str):
		return value
	for key in ("stdout", "output", "console", "text"):
		value = payload.get(key)
		if isinstance(value, str):
			return value
	for key in ("result", "task"):
		value = payload.get(key)
		if isinstance(value, dict):
			text = _result_text(value)
			if text:
				return text
	return ""


def _result_metadata(payload: dict[str, Any]) -> dict[str, Any]:
	for value in (payload.get("result"), payload.get("task"), payload):
		if isinstance(value, dict):
			if any(key in value for key in ("returncode", "timed_out", "error")):
				return value
			nested = value.get("result")
			if isinstance(nested, dict) and any(key in nested for key in ("returncode", "timed_out", "error")):
				return nested
	return {}


class NeuSbox:
	"""Submit and monitor only finite tasks created by this coordinator."""

	def __init__(
		self,
		*,
		local_host: str | None = None,
		run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
		sleep: Callable[[float], None] = time.sleep,
		clock: Callable[[], float] = time.monotonic,
		poll_interval: float = 1.0,
		command_timeout: float = 15.0,
	) -> None:
		self.local_host = local_host
		self._run = run
		self._sleep = sleep
		self._clock = clock
		self.poll_interval = poll_interval
		if command_timeout <= 0 or command_timeout > 60:
			raise ValueError("Scheduler command timeouts must be between 0 and 60 seconds")
		self.command_timeout = command_timeout
		self._owned: dict[tuple[str | None, str], Allocation] = {}

	def _command(self, host: str | None, args: list[str]) -> list[str]:
		command = ["neu-sbox", "--json", *args]
		if host is None or host == self.local_host:
			return command
		return [
			"ssh",
			"-o",
			"BatchMode=yes",
			"-o",
			"ConnectTimeout=5",
			host,
			shlex.join(command),
		]

	def _call(self, host: str | None, args: list[str], *, timeout: float | None = None) -> dict[str, Any]:
		command = self._command(host, args)
		deadline = self.command_timeout if timeout is None else min(self.command_timeout, timeout)
		if deadline <= 0:
			raise TimeoutError(f"Scheduler command deadline expired on {host or 'local'}")
		try:
			completed = self._run(command, capture_output=True, text=True, check=False, timeout=deadline)
		except subprocess.TimeoutExpired as error:
			raise TimeoutError(
				f"Scheduler command timed out on {host or 'local'} after {deadline:g}s: {command}"
			) from error
		except OSError as error:
			raise SchedulerError(f"scheduler command failed to start on {host or 'local'}: {error}") from error
		if completed.returncode != 0:
			raise SchedulerError(
				f"scheduler command failed on {host or 'local'} with {completed.returncode}: {completed.stderr.strip()}"
			)
		try:
			payload = json.loads(completed.stdout)
		except json.JSONDecodeError as error:
			raise SchedulerError(
				f"scheduler returned invalid JSON on {host or 'local'}: {completed.stdout.strip()}"
			) from error
		if not isinstance(payload, dict):
			raise SchedulerError("scheduler returned a non-object JSON response")
		return payload

	def submit(
		self,
		host: str | None,
		request: ResourceRequest,
		command: str,
		*,
		cwd: Path,
		wall_seconds: float,
		ledger: Path,
		label: str,
	) -> Allocation:
		if wall_seconds <= 0:
			raise ValueError("Peer worker wall time must be positive")
		ledger.parent.mkdir(parents=True, exist_ok=True)
		wrapped = (
			f"cd {shlex.quote(str(cwd))} && exec timeout --foreground --kill-after=30s "
			f"{wall_seconds:g} bash -lc {shlex.quote(command)}"
		)
		payload = self._call(
			host,
			[
				"submit",
				"--cpu",
				str(request.cpu),
				"--mem",
				str(request.memory_gib),
				"--device-num",
				str(request.gpus),
				"--",
				"bash",
				"-lc",
				wrapped,
			],
		)
		task_id = payload.get("task_id")
		if not isinstance(task_id, str) or not task_id:
			raise SchedulerError(f"scheduler submission did not return a task_id on {host or 'local'}")
		allocation = Allocation(host, task_id, request, command)
		self._owned[(host, task_id)] = allocation
		try:
			ledger.write_text(
				json.dumps(
					{
						"task_id": task_id,
						"host": host or self.local_host or "local",
						"label": label,
						"request": request.__dict__,
						"command": command,
					},
					indent=2,
				)
				+ "\n"
			)
		except BaseException:
			self.cancel(allocation)
			raise
		return allocation

	def current_allocation(self) -> tuple[dict[str, Any] | None, list[str]]:
		"""Return this process's sandbox record and granted GPU UUIDs, if allocated."""
		from scripts.peer.allocation import resolve_current_allocation

		return resolve_current_allocation(run=self._run, timeout=self.command_timeout)

	def snapshot(self, task_id: str, *, host: str | None = None) -> dict[str, Any]:
		"""Read the current snapshot for one exact task ID."""
		if not task_id:
			raise ValueError("A task ID is required")
		return self._call(host, ["result", task_id])

	def status(self, allocation: Allocation) -> dict[str, Any]:
		payload = self.snapshot(allocation.task_id, host=allocation.host)
		allocation.last_status = payload
		allocation.state = _state(payload)
		return payload

	def result(self, allocation: Allocation) -> tuple[str, dict[str, Any]]:
		payload = self.snapshot(allocation.task_id, host=allocation.host)
		state = _state(payload)
		metadata = _result_metadata(payload)
		if state != "completed":
			raise SchedulerError(
				f"owned task {allocation.task_id} did not complete successfully (state={state}): {payload}"
			)
		if not metadata:
			raise SchedulerError(f"owned task {allocation.task_id} result omitted return-code metadata: {payload}")
		if metadata.get("timed_out") is True:
			raise SchedulerError(f"owned task {allocation.task_id} exceeded its worker time limit: {payload}")
		returncode = metadata.get("returncode")
		if type(returncode) is not int or returncode != 0:
			raise SchedulerError(f"owned task {allocation.task_id} returned {returncode!r}: {payload}")
		if metadata.get("error") not in (None, ""):
			raise SchedulerError(f"owned task {allocation.task_id} reported an execution error: {metadata['error']}")
		return _result_text(payload), payload

	def wait_terminal(self, task_id: str, *, host: str | None = None, max_seconds: float = 3600) -> dict[str, Any]:
		"""Poll one task snapshot until a terminal state, without changing the task."""
		if max_seconds <= 0 or max_seconds > 3600:
			raise ValueError("Peer scheduler monitor windows must be between 0 and 3600 seconds")
		deadline = self._clock() + max_seconds
		while True:
			remaining = deadline - self._clock()
			if remaining <= 0:
				raise TimeoutError(f"Timed out waiting for task {task_id} to finish")
			payload = self._call(host, ["result", task_id], timeout=min(self.command_timeout, remaining))
			if _state(payload) in TERMINAL_STATES:
				return payload
			remaining = deadline - self._clock()
			if remaining <= 0:
				raise TimeoutError(f"Timed out waiting for task {task_id} to finish")
			self._sleep(min(self.poll_interval, remaining))

	def _monitor(
		self,
		allocations: Iterable[Allocation],
		*,
		target: str,
		max_seconds: float,
	) -> None:
		items = tuple(allocations)
		if not items:
			return
		if max_seconds <= 0 or max_seconds > 3600:
			raise ValueError("Peer scheduler monitor windows must be between 0 and 3600 seconds")
		deadline = self._clock() + max_seconds
		while True:
			states: list[str] = []
			for allocation in items:
				remaining = deadline - self._clock()
				if remaining <= 0:
					raise TimeoutError(f"Timed out waiting for peer tasks to become {target}: {states}")
				payload = self._call(
					allocation.host,
					["result", allocation.task_id],
					timeout=min(self.command_timeout, remaining),
				)
				allocation.last_status = payload
				allocation.state = _state(payload)
				state = allocation.state
				states.append(state)
				if target == "running" and state in TERMINAL_STATES and state != "completed":
					raise SchedulerError(
						f"owned task {allocation.task_id} on {allocation.host or 'local'} entered {state}: {payload}"
					)
				if target == "running" and state == "completed":
					raise SchedulerError(
						f"owned task {allocation.task_id} completed before its peer allocation was running"
					)
				if target == "completed" and state in {
					"failed",
					"cancelled",
					"canceled",
					"timeout",
					"timed_out",
					"error",
				}:
					raise SchedulerError(
						f"owned task {allocation.task_id} on {allocation.host or 'local'} entered {state}: {payload}"
					)
			if all(state in (RUNNING_STATES if target == "running" else {"completed"}) for state in states):
				return
			remaining = deadline - self._clock()
			if remaining <= 0:
				raise TimeoutError(f"Timed out waiting for peer tasks to become {target}: {states}")
			self._sleep(min(self.poll_interval, remaining))

	def wait_running(self, allocations: Iterable[Allocation], max_seconds: float) -> None:
		self._monitor(allocations, target="running", max_seconds=max_seconds)

	def wait_completed(self, allocations: Iterable[Allocation], max_seconds: float) -> None:
		self._monitor(allocations, target="completed", max_seconds=max_seconds)

	def cancel(self, allocation: Allocation) -> None:
		key = (allocation.host, allocation.task_id)
		if self._owned.get(key) is not allocation:
			raise SchedulerError(f"refusing to cancel task not submitted by this coordinator: {allocation.task_id}")
		if allocation.state in TERMINAL_STATES:
			self._owned.pop(key, None)
			return
		command = ["python3", "-c", _DELETE_TASK_SCRIPT, allocation.task_id]
		if allocation.host is not None and allocation.host != self.local_host:
			remote = shlex.join(command)
			command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", allocation.host, remote]
		try:
			completed = self._run(
				command,
				capture_output=True,
				text=True,
				check=False,
				timeout=min(self.command_timeout, 15.0),
			)
		except subprocess.TimeoutExpired as error:
			raise TimeoutError(f"Owned-task cancellation timed out for {allocation.task_id}") from error
		except OSError as error:
			raise SchedulerError(
				f"Owned-task cancellation failed to start for {allocation.task_id}: {error}"
			) from error
		if completed.returncode != 0:
			raise SchedulerError(
				f"Owned-task cancellation failed for {allocation.task_id} with {completed.returncode}: "
				f"{completed.stderr.strip()}"
			)
		allocation.state = "cancelled"
		self._owned.pop(key, None)

	def cleanup(self, allocations: Iterable[Allocation]) -> list[str]:
		errors = []
		for allocation in allocations:
			try:
				self.cancel(allocation)
			except Exception as error:
				errors.append(f"{allocation.task_id}: {error}")
		return errors


def create_app():
	"""Build read-only task inspection commands without a global CLI dependency."""
	import typer

	app = typer.Typer(no_args_is_help=True, help="Inspect and monitor Neu Box tasks without mutating them.")

	@app.command("result")
	def result(task_id: str, host: str | None = typer.Option(None, help="SSH host running the Neu Box worker")) -> None:
		try:
			payload = NeuSbox().snapshot(task_id, host=host)
		except (SchedulerError, TimeoutError) as error:
			typer.echo(str(error), err=True)
			raise typer.Exit(1) from error
		typer.echo(json.dumps(payload, indent=2))
		if _state(payload) in TERMINAL_STATES and _state(payload) != "completed":
			raise typer.Exit(1)

	@app.command("wait")
	def wait(
		task_id: str,
		host: str | None = typer.Option(None, help="SSH host running the Neu Box worker"),
		monitor_window: float = typer.Option(3600, min=1, max=3600, help="Maximum wait in seconds"),
		poll_interval: float = typer.Option(2, min=0.1, help="Seconds between result snapshots"),
	) -> None:
		try:
			payload = NeuSbox(poll_interval=poll_interval).wait_terminal(
				task_id,
				host=host,
				max_seconds=monitor_window,
			)
		except (SchedulerError, TimeoutError) as error:
			typer.echo(str(error), err=True)
			raise typer.Exit(1) from error
		typer.echo(json.dumps(payload, indent=2))
		if _state(payload) != "completed":
			raise typer.Exit(1)

	return app


def cli() -> None:
	create_app()()


if __name__ == "__main__":
	cli()
