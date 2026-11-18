import json
import subprocess
import sys
from pathlib import Path

import dgl
import pytest
import torch
from typer.testing import CliRunner

from benchmarks import external
from benchmarks.sources import PinnedSource
from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire
from scripts.evaluate.external import app


def _checkout(tmp_path):
	checkout = tmp_path / "checkout"
	checkout.mkdir()
	subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
	(checkout / "source.py").write_text("VALUE = 1\n")
	subprocess.run(["git", "add", "source.py"], cwd=checkout, check=True)
	subprocess.run(
		["git", "-c", "user.name=External", "-c", "user.email=external@example.invalid", "commit", "-qm", "fixture"],
		cwd=checkout,
		check=True,
	)
	revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
	return checkout, PinnedSource("ripple", revision, "https://example.invalid/ripple")


def _fake_python(tmp_path, mode="pass"):
	path = tmp_path / "fake-python"
	lines = [
		"#!/usr/bin/env python3",
		"import json, sys, time",
		"from pathlib import Path",
		"args = sys.argv",
		"output = Path(args[args.index('--output') + 1])",
	]
	if mode == "timeout":
		lines.append("time.sleep(10)")
	else:
		lines.extend(
			[
				"output.mkdir()",
				f"report = {{'valid': {mode != 'invalid'!r}, 'dependencies': {{'python': 'fake'}}}}",
				"output.joinpath('verification.json').write_text(json.dumps(report))",
			]
		)
		if mode == "fail-valid":
			lines.append("raise SystemExit(7)")
	path.write_text("\n".join(lines) + "\n")
	path.chmod(0o755)
	return path


def _fake_python_with_descendant(tmp_path, pid_file):
	path = tmp_path / "fake-python-descendant"
	path.write_text(
		"\n".join(
			[
				"#!/usr/bin/env python3",
				"import json, subprocess, sys",
				"from pathlib import Path",
				f"pid_file = Path({str(pid_file)!r})",
				"child = subprocess.Popen(['sleep', '30'])",
				"pid_file.write_text(str(child.pid))",
				"output = Path(sys.argv[sys.argv.index('--output') + 1])",
				"output.mkdir()",
				"output.joinpath('verification.json').write_text(json.dumps({'valid': True}))",
			]
		)
		+ "\n"
	)
	path.chmod(0o755)
	return path


def test_capability_manifest_does_not_invent_performance_values():
	manifest = external.capability_manifest("ripple")
	assert manifest["performance_replay"] is True
	assert manifest["timing_available"] is True
	assert manifest["replay_gate"] == "gc_mean_zero_self_checkpoint_and_recorded_workload"
	assert manifest["adapter_supported"]["updates"] == ["insert", "delete", "feature", "noop"]
	helios = external.capability_manifest("helios")
	assert helios["performance_replay"] is False
	assert helios["numerical_verification"] is False
	assert "timestamp-top-k" in helios["performance_blocker"]


def test_runner_passes_pinned_source_and_publishes_logs_without_overwrite(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	python_executable = _fake_python(tmp_path)
	output = tmp_path / "result"
	report = external.run_external(
		"ripple",
		checkout,
		output,
		python_executable=python_executable,
		hardware_profile="fixture-hardware",
		timeout=5,
	)
	assert report["status"] == "passed"
	assert report["source"]["revision"] == pin.revision
	assert report["python_executable"] == str(python_executable.resolve())
	assert len(report["python_executable_sha256"]) == 64
	assert report["hardware_profile_enforced"] is False
	assert report["dependency_identity"] == {"python": "fake"}
	assert (output / "external.stdout.log").exists()
	assert json.loads((output / "external-run.json").read_text())["status"] == "passed"
	with pytest.raises(ValueError, match="refusing to overwrite"):
		external.run_external("ripple", checkout, output, python_executable=python_executable)


def test_runner_timeout_is_bounded_and_publishes_timeout_state(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	output = tmp_path / "timeout"
	report = external.run_external(
		"ripple",
		checkout,
		output,
		python_executable=_fake_python(tmp_path, "timeout"),
		timeout=0.1,
	)
	assert report["status"] == "timeout"
	assert report["returncode"] == 124
	assert json.loads((output / "external-run.json").read_text())["status"] == "timeout"


def test_runner_reports_exact_path_and_rejects_missing_requested_python(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	output = tmp_path / "bad-python"
	report = external.run_external("ripple", checkout, output, python_executable=tmp_path / "missing-python", timeout=5)
	assert report["status"] == "failed"
	assert report["failure_phase"] == "python_preflight"
	assert "missing" in report["failure"].lower()


def test_runner_does_not_promote_nonzero_child_with_valid_report(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	report = external.run_external(
		"ripple",
		checkout,
		tmp_path / "child-failure",
		python_executable=_fake_python(tmp_path, "fail-valid"),
		timeout=5,
	)
	assert report["status"] == "failed"
	assert report["returncode"] == 7


def test_runner_preserves_validation_failed_as_nonzero_status(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	report = external.run_external(
		"ripple",
		checkout,
		tmp_path / "validation-failure",
		python_executable=_fake_python(tmp_path, "invalid"),
		timeout=5,
	)
	assert report["status"] == "validation_failed"
	assert report["returncode"] == 0


def test_runner_cleans_owned_descendant_after_normal_leader_exit(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	pid_file = tmp_path / "child.pid"
	report = external.run_external(
		"ripple",
		checkout,
		tmp_path / "descendant",
		python_executable=_fake_python_with_descendant(tmp_path, pid_file),
		timeout=5,
	)
	assert report["status"] == "passed"
	child_pid = int(pid_file.read_text())
	with pytest.raises(ProcessLookupError):
		__import__("os").kill(child_pid, 0)


def test_runner_bounds_git_preflight_and_keeps_timeout_state(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	monkeypatch.setattr(
		external,
		"source_identity",
		lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 1)),
	)
	report = external.run_external(
		"ripple", checkout, tmp_path / "preflight-timeout", python_executable=_fake_python(tmp_path), timeout=1
	)
	assert report["status"] == "timeout"
	assert report["failure_phase"] == "source_preflight"
	assert report["returncode"] == 124


def test_runner_rejects_dirty_source_before_launch(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	(checkout / "untracked.txt").write_text("dirty\n")
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	report = external.run_external(
		"ripple", checkout, tmp_path / "dirty", python_executable=_fake_python(tmp_path), timeout=5
	)
	assert report["status"] == "failed"
	assert report["failure_phase"] == "source_preflight"


def test_performance_mode_is_explicitly_unsupported(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	report = external.run_external(
		"ripple",
		checkout,
		tmp_path / "performance",
		python_executable=Path("python"),
		mode="performance",
	)
	assert report["status"] == "unsupported"
	assert "supported modes" in report["failure"].lower()


def test_helios_source_pin_is_reported_but_never_promoted_to_a_runner(tmp_path, monkeypatch):
	checkout, pin = _checkout(tmp_path)
	monkeypatch.setattr(external.PinnedSource, "load", classmethod(lambda cls, name: pin))
	monkeypatch.setattr(
		external,
		"source_identity",
		lambda name, path, timeout=30.0: {"name": name, "revision": pin.revision, "checkout": str(path)},
	)
	report = external.run_external(
		"helios", checkout, tmp_path / "helios", python_executable=_fake_python(tmp_path), mode="replay", timeout=5
	)
	assert report["status"] == "unsupported"
	assert report["failure_phase"] == "compatibility_gate"
	assert report["source"]["revision"] == pin.revision
	assert "timestamp-top-k" in report["failure"]
	assert "command" not in report
	assert json.loads((tmp_path / "helios" / "external-run.json").read_text())["status"] == "unsupported"


def test_cli_help_exposes_one_external_entrypoint():
	result = CliRunner().invoke(app, ["--help"])
	assert result.exit_code == 0
	assert "--artifact" in result.stdout


@pytest.mark.skipif(not Path("/tmp/neutronrt-paper-ripple").is_dir(), reason="pinned Ripple checkout is unavailable")
@pytest.mark.parametrize("processor_name", ["incremental", "rc"])
def test_ripple_replay_runs_generated_workload_through_author_processor(tmp_path, processor_name):
	graph = dgl.graph(([0, 1, 0], [1, 2, 3]), num_nodes=4)
	features = torch.tensor(
		[[1.0, -1.0, 0.5, 2.0], [0.2, 0.3, -0.4, 0.8], [-1.0, 0.5, 0.7, -0.2], [0.1, 0.0, -0.1, 0.2]]
	)
	graph.ndata["feat"] = features
	model = ModelSpec("sage", 4, 4, 2, layers=2).build()
	checkpoint = tmp_path / "gcmean.pt"
	torch.save(
		{
			"model_config": {
				"model_name": "sage",
				"num_features": 4,
				"num_hiddens": 4,
				"num_classes": 2,
				"num_layers": 2,
			},
			"state_dict": model.state_dict(),
		},
		checkpoint,
	)
	spec = ModelSpec("sage", 4, 4, 2, layers=2, checkpoint=str(checkpoint))
	request = wire.Update(
		kind=wire.Update.INSERT,
		first=_to_wire(torch.tensor([2])),
		second=_to_wire(torch.tensor([3])),
		request_id="insert-0",
	)
	workload = RecordedWorkload(
		spec,
		inference_identity(graph, features, spec),
		"test-source",
		(),
		((request, _to_wire(torch.tensor([0, 1, 2, 3]))),),
		torch.arange(4),
		"all",
		71,
		0,
		1e-6,
		1e-5,
	)
	workload_path = tmp_path / "workload.pt"
	workload.save(workload_path)
	graph_path = tmp_path / "graph.bin"
	dgl.save_graphs(str(graph_path), [graph])
	report = external.run_external(
		"ripple",
		Path("/tmp/neutronrt-paper-ripple"),
		tmp_path / "replay",
		python_executable=Path(sys.executable),
		timeout=30,
		mode="replay",
		replay_options={
			"workload": workload_path,
			"graph": graph_path,
			"checkpoint": checkpoint,
			"processor": processor_name,
			"gcmean_compatibility": True,
		},
	)
	assert report["status"] == "passed", report
	assert report["verification"]["replay"]["processor"] == processor_name
	verification = json.loads((tmp_path / "replay" / "verification.json").read_text())
	assert verification["timing"]["metrics_valid"] is True
	assert verification["timing"]["throughput_events_per_second"] is not None
	assert verification["timing"]["throughput_batches_per_second"] is not None
	assert verification["timing"]["full_harness_elapsed_seconds"] >= verification["timing"]["service_elapsed_seconds"]
	assert verification["counts"]["input_event_count"] == 1
	assert verification["initialization"]["processor_bootstrap_seconds"] >= 0
	assert verification["gcmean_compatibility"]["mode"] == "explicit_zero_self_and_bias_compatibility"
	assert (
		verification["gcmean_compatibility"]["original_trace_digest"]
		== verification["gcmean_compatibility"]["adapted_trace_digest"]
	)
