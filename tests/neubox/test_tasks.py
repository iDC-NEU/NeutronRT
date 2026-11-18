import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from scripts.coding.run import command
from scripts.neubox.tasks import call, entries, main


def test_tasks_use_structured_fields_and_preserve_multiline_commands():
	row = {
		"task_id": "abc",
		"status": "running",
		"user_id": "lhr",
		"command": "python\nscript.py",
		"cpu": 4,
		"mem": "8G",
		"devices": [],
	}
	for key in ("queue", "tasks"):
		result = entries({key: [row]})
		assert result[0].hash == "abc" and result[0].command == "python\nscript.py"
		assert "RAM=8G" in str(result[0])


def test_profile_session_is_persistent_and_streamed():
	args = command("lab", Path("/tmp/work"), Path("/tmp/result.md"), None)
	assert args[args.index("--profile") + 1] == "lab"
	assert "--json" in args and "--ephemeral" not in args
	assert args[-1] == "-"
	resume = command("lab", Path("/tmp/work"), Path("/tmp/result.md"), "session-id")
	assert resume[-3:] == ["resume", "session-id", "-"]


def test_task_command_errors_include_scheduler_diagnostics(monkeypatch):
	monkeypatch.setattr(
		"scripts.neubox.tasks.subprocess.run",
		lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="scheduler unavailable"),
	)
	with pytest.raises(RuntimeError, match="scheduler unavailable"):
		call("tasks")


def test_watch_requires_completed_zero_returncode(monkeypatch, capsys):
	monkeypatch.setattr(
		"scripts.neubox.tasks.call",
		lambda *args: {
			"task_id": "task-1",
			"status": "failed",
			"result": {"returncode": 0},
			"log": "failed after reporting a zero process code",
		},
	)
	with pytest.raises(typer.Exit) as error:
		main(hash=["task-1"], watch=True, seconds=1, json_output=True)
	assert error.value.exit_code == 1
	assert json.loads(capsys.readouterr().out)[0]["status"] == "failed"
