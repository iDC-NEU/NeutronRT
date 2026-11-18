"""Queue a session notification only after recording its durable receipt."""

from pathlib import Path

from scripts.neubox.watch import _atomic_write, _queue_callback


def notify(output: Path, thread: str, message: str, event: str) -> dict:
	output.mkdir(parents=True, exist_ok=False)
	receipt = {"event": event, "monitor_exit": 0, "callback_status": "pending"}
	path = output / "receipt.json"
	_atomic_write(path, receipt)
	_queue_callback(path, output, [], thread, receipt, message)
	_atomic_write(path, receipt)
	return receipt


def main(output: Path, thread: str, message: str, event: str = "notification"):
	"""Write and queue a notification with an auditable receipt."""
	import json

	import typer

	typer.echo(json.dumps(notify(output, thread, message, event)))


if __name__ == "__main__":
	import typer

	typer.run(main)
