"""Filter, inspect and wait for Neu Box tasks using its structured CLI output."""

import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Annotated

import typer

STATUS_COLORS = {
	"queued": typer.colors.YELLOW,
	"pending": typer.colors.YELLOW,
	"running": typer.colors.GREEN,
	"completed": typer.colors.CYAN,
	"failed": typer.colors.RED,
	"cancelled": typer.colors.BRIGHT_BLACK,
	"canceled": typer.colors.BRIGHT_BLACK,
}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}


@dataclass
class Entry:
	hash: str
	status: str
	user: str
	command: str
	resources: str

	def __str__(self):
		return f"{self.hash} {self.status} {self.user} {self.resources}\n\t{self.command}"

	def styled(self):
		identifier = typer.style(self.hash, fg=typer.colors.CYAN, bold=True)
		status = typer.style(self.status, fg=STATUS_COLORS.get(self.status), bold=True)
		user = typer.style(self.user, fg=typer.colors.MAGENTA)
		return f"{identifier} {status} {user} {self.resources}\n\t{self.command}"


def call(*args: str) -> dict:
	command = ["neu-sbox", "--json", *args]
	try:
		result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
	except OSError as error:
		raise RuntimeError(f"Could not run {' '.join(command)}: {error}") from error
	if result.returncode:
		detail = (result.stderr or result.stdout).strip()
		raise RuntimeError(f"Neu Box command failed with exit {result.returncode}: {detail or 'no diagnostic output'}")
	try:
		payload = json.loads(result.stdout)
	except json.JSONDecodeError as error:
		raise RuntimeError(f"Neu Box returned invalid JSON: {error}") from error
	if not isinstance(payload, dict):
		raise RuntimeError("Neu Box returned a JSON value instead of an object")
	return payload


def entries(payload: dict) -> list[Entry]:
	rows = payload.get("queue", payload.get("tasks"))
	if not isinstance(rows, list):
		raise ValueError("Task response has no queue/tasks array")
	return [
		Entry(
			str(row["task_id"]),
			row["status"],
			row.get("user_id", ""),
			row.get("command", ""),
			f"CPU={row.get('cpu')} RAM={row.get('mem')} GPUs={row.get('devices', [])}",
		)
		for row in rows
	]


def _successful_completion(row: dict) -> bool:
	return row.get("status") == "completed" and (row.get("result") or {}).get("returncode") == 0


def main(
	topk: Annotated[int, typer.Option("--topk", "-k", min=-1)] = -1,
	hash: Annotated[list[str] | None, typer.Option("--hash", "--task-id")] = None,
	status: str | None = None,
	user: Annotated[str | None, typer.Option("--user", "-u")] = None,
	command: str | None = None,
	health: bool = False,
	watch: bool = False,
	seconds: Annotated[float, typer.Option(min=0.001, max=3600)] = 3600,
	interval: Annotated[float, typer.Option(min=1, max=60)] = 15,
	json_output: Annotated[bool, typer.Option("--json")] = False,
	color: Annotated[bool | None, typer.Option("--color/--no-color", help="Default: color in terminals only.")] = None,
):
	"""Watch requires explicit task IDs; a monitor timeout never cancels remote work."""
	if watch and not hash:
		raise typer.BadParameter("--watch requires explicit --task-id values")
	try:
		pattern = re.compile(command) if command else None
	except re.error as error:
		raise typer.BadParameter(str(error)) from error
	deadline = time.monotonic() + seconds
	previous = None
	while True:
		if watch or health and hash:
			rows = [call("result", task) for task in hash or []]
		else:
			rows = [vars(row) for row in entries(call("tasks"))]
			rows = [
				row
				for row in rows
				if (not hash or row["hash"] in hash)
				and (not status or row["status"] == status)
				and (not user or row["user"] == user)
				and (not pattern or pattern.search(row["command"]))
			]
			if topk >= 0:
				rows = rows[:topk]
			if health:
				rows = [call("result", row["hash"]) for row in rows]
		if health or watch:
			rows = [
				{
					**{k: row.get(k) for k in ("task_id", "status", "cpu", "mem", "devices", "result")},
					"log_tail": row.get("log", "")[-1500:],
				}
				for row in rows
			]
		encoded = json.dumps(rows, ensure_ascii=False)
		if encoded != previous:
			if json_output:
				typer.echo(encoded, color=False)
			else:
				for row in rows:
					if health or watch:
						entry = Entry(
							row["task_id"],
							row["status"],
							"",
							row["log_tail"],
							f"CPU={row['cpu']} RAM={row['mem']} GPUs={row['devices']} result={row['result']}",
						)
					else:
						entry = Entry(**row)
					typer.echo(entry.styled(), color=color)
			previous = encoded
		if not watch:
			return
		if not rows:
			raise RuntimeError("No Neu Box task results were returned")
		if all(row["status"] in TERMINAL_STATUSES for row in rows):
			raise typer.Exit(0 if all(_successful_completion(row) for row in rows) else 1)
		remaining = deadline - time.monotonic()
		if remaining <= 0:
			typer.echo("Monitor timed out; remote tasks continue. Check health before renewing.", err=True)
			raise typer.Exit(124)
		time.sleep(min(interval, remaining))


if __name__ == "__main__":
	typer.run(main)
