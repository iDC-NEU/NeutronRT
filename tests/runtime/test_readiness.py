import json
import os
from types import SimpleNamespace

import pytest

from scripts.simple import write_ready
from scripts.validate.inference_rpc import wait_ready


def test_startup_notification_reports_actual_bound_port_and_only_matches_its_child(tmp_path):
	path = tmp_path / "server.ready.json"
	write_ready(path, "127.0.0.1:0", 12345, "expected-session")
	process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
	assert wait_ready(process, path) == {
		"pid": os.getpid(),
		"address": "127.0.0.1:12345",
		"session": "expected-session",
	}
	assert [value.name for value in tmp_path.iterdir()] == [path.name]
	path.write_text(json.dumps({"pid": os.getpid() + 1, "session": "other-server"}))
	with pytest.raises(TimeoutError):
		wait_ready(process, path, timeout=0.001)
	with pytest.raises(RuntimeError, match="exited"):
		wait_ready(SimpleNamespace(pid=os.getpid(), poll=lambda: 1), path)
