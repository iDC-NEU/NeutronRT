import json

import dgl
import torch

from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire
from scripts.evaluate import external_cohort


def _wire(tensor):
	return _to_wire(tensor)


def _workload(path, repetition):
	graph = dgl.graph(([0, 1, 2, 0], [1, 2, 0, 3]), num_nodes=4)
	features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
	graph.ndata["feat"] = features
	spec = ModelSpec("sage", 4, 4, 2, layers=2, seed=71)
	steps = []
	for index in range(20):
		if index < 7:
			request = wire.Update(
				kind=wire.Update.FEATURES,
				first=_wire(torch.tensor([index % 4], dtype=torch.int64)),
				second=_wire(torch.full((1, 4), float(index))),
			)
		elif index < 14:
			request = wire.Update(
				kind=wire.Update.INSERT,
				first=_wire(torch.tensor([index % 4], dtype=torch.int64)),
				second=_wire(torch.tensor([(index + 1) % 4], dtype=torch.int64)),
			)
		else:
			request = wire.Update(
				kind=wire.Update.DELETE,
				first=_wire(torch.tensor([index % 4], dtype=torch.int64)),
				second=_wire(torch.tensor([(index + 1) % 4], dtype=torch.int64)),
			)
		steps.append((request, _wire(torch.tensor([0], dtype=torch.int64))))
	workload = RecordedWorkload(
		spec,
		inference_identity(graph, features, spec),
		"fixture",
		(),
		tuple(steps),
		torch.arange(4),
		"all",
		71 + repetition,
		repetition,
		1e-6,
		1e-5,
	)
	workload.save(path)
	return workload


def test_external_cohort_runs_all_trials_and_preserves_failure_states(tmp_path, monkeypatch):
	workloads = [tmp_path / f"trace-{repetition}.pt" for repetition in range(5)]
	workload_values = [_workload(path, repetition) for repetition, path in enumerate(workloads)]
	graph_path = tmp_path / "graph.bin"
	dgl.save_graphs(str(graph_path), [dgl.graph(([0, 1], [1, 2]), num_nodes=4)])
	trials = [
		{
			"id": f"{dataset}-r{repetition}",
			"dataset": dataset,
			"repetition": repetition,
			"trace_seed": 71 + repetition,
			"trace_digest": workload_values[repetition].digest,
			"workload": str(workloads[repetition]),
			"graph": str(graph_path),
		}
		for dataset in ("arxiv", "products")
		for repetition in range(5)
	]
	plan_path = tmp_path / "plan.json"
	plan_path.write_text(
		json.dumps(
			{
				"schema": 1,
				"cohort_kind": "planned",
				"id": "fixture",
				"artifact": "ripple",
				"display_name": "Ripple GC-mean",
				"processor": "incremental",
				"source_checkout": str(tmp_path),
				"python_executable": "/usr/bin/python3",
				"hardware_profile": "external-cpu",
				"trial_timeout_seconds": 5,
				"checkpoint": None,
				"workload_contract": {
					"update_requests": 20,
					"updates": {
						"features": {"requests": 7, "changes_per_request": 1},
						"insert": {"requests": 7, "changes_per_request": 1},
						"delete": {"requests": 6, "changes_per_request": 1},
					},
					"query_requests": 20,
					"query_rows": 20,
				},
				"trials": trials,
			},
			indent=2,
		)
	)
	call_count = 0

	def fake_run_external(name, checkout, output, **kwargs):
		nonlocal call_count
		call_count += 1
		output.mkdir(parents=True)
		valid = call_count != 2
		report = {
			"valid": valid,
			"correctness": [],
			"replay": {"oracle_validation": "sampled fixture"},
			"timing": {"metrics_valid": True, "service_elapsed_seconds": 0.1},
		}
		(output / "verification.json").write_text(json.dumps(report))
		return {"status": "passed" if valid else "validation_failed", "failure": None if valid else "fixture mismatch"}

	monkeypatch.setattr(external_cohort, "run_external", fake_run_external)
	result = external_cohort.run(plan_path, tmp_path / "cohort")
	assert call_count == 10
	assert result["completed_trials"] == 10
	assert result["passed_trials"] == 9
	assert result["valid_performance_trials"] == 10
	assert result["trials"][1]["status"] == "validation_failed"
	assert (tmp_path / "cohort/trials/arxiv-r0/process-status.json").is_file()
	assert json.loads((tmp_path / "cohort/summary.json").read_text())["completed_trials"] == 10
