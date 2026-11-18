"""Bind lightweight monitors to the owning Codex process without touching jobs."""

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionOwner:
	pid: int
	started: str
	thread: str

	@classmethod
	def discover(cls, thread: str) -> "SessionOwner":
		pid = os.getpid()
		while pid > 1:
			folder = Path(f"/proc/{pid}")
			fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
			command = (folder / "cmdline").read_bytes().split(b"\0")
			if command and Path(os.fsdecode(command[0])).name == "codex" and thread.encode() in command:
				return cls(pid, fields[19], thread)
			pid = int(fields[1])
		raise RuntimeError("Cannot identify this thread's Codex owner process; refusing an orphan monitor")

	def alive(self) -> bool:
		try:
			fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
			return fields[0] != "Z" and fields[19] == self.started
		except (OSError, IndexError):
			return False

	def wait(self, seconds: float) -> bool:
		"""Return false promptly if the session ends, including PID reuse."""
		deadline = time.monotonic() + seconds
		while self.alive():
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				return True
			time.sleep(min(1, remaining))
		return False

	def record(self) -> dict:
		return asdict(self)


def main(thread: str | None = None):
	"""Print the owner identity used to scope this session's monitors."""
	import json

	import typer

	thread = thread or os.environ.get("CODEX_THREAD_ID")
	if not thread:
		raise typer.BadParameter("Provide --thread or CODEX_THREAD_ID")
	typer.echo(json.dumps(SessionOwner.discover(thread).record()))


if __name__ == "__main__":
	import typer

	typer.run(main)
