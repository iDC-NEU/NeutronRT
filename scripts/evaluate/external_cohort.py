"""Run a frozen, serial cohort of pinned external baseline replays."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.external import run_external
from lib.profiling.workload import RecordedWorkload


def _save(path: Path, value: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
		temporary = Path(stream.name)
		json.dump(value, stream, indent=2, sort_keys=True)
		stream.write("\n")
		stream.flush()
		os.fsync(stream.fileno())
	temporary.replace(path)


def _validate_plan(plan: dict) -> list[dict]:
	trials = plan.get("trials")
	if plan.get("schema") != 1 or not isinstance(trials, list) or not trials:
		raise ValueError("External cohort plan must use schema 1 and contain trials")
	kind = plan.get("cohort_kind", "planned")
	if kind not in {"smoke", "planned"}:
		raise ValueError("cohort_kind must be smoke or planned")
	if plan.get("artifact") != "ripple" or plan.get("processor") != "incremental":
		raise ValueError("This runner currently admits only the pinned Ripple incremental processor")
	if not 0 < plan.get("trial_timeout_seconds", 0) <= 3600:
		raise ValueError("Each external trial timeout must be in (0, 3600]")
	ids = [trial.get("id") for trial in trials]
	if any(not isinstance(value, str) or not value or Path(value).name != value for value in ids):
		raise ValueError("Trial IDs must be nonempty directory names")
	if len(set(ids)) != len(ids):
		raise ValueError("Trial IDs must be unique")
	if kind == "planned" and len(trials) != 10:
		raise ValueError("The planned overview cohort requires five paired repetitions for arXiv and Products")
	if kind == "smoke" and len(trials) != 1:
		raise ValueError("A smoke cohort must contain exactly one labeled trial")
	for dataset in ("arxiv", "products"):
		group = [trial for trial in trials if trial.get("dataset") == dataset]
		if kind == "planned" and group and {trial.get("repetition") for trial in group} != set(range(5)):
			raise ValueError(f"Expected five repetitions for {dataset}")
		for trial in group:
			for key in ("workload", "graph", "trace_digest"):
				if not isinstance(trial.get(key), str) or not trial[key]:
					raise ValueError(f"Trial {trial['id']} is missing {key}")
			if not Path(trial["workload"]).is_file() or not Path(trial["graph"]).exists():
				raise FileNotFoundError(f"Trial input is missing: {trial['id']}")
	return trials


def _validate_workload_contract(counts: dict, contract: dict, trial_id: str) -> None:
	if counts["update_requests"] != contract["update_requests"]:
		raise ValueError(f"Update request count differs from the frozen contract: {trial_id}")
	for kind, expected in contract["updates"].items():
		actual = counts["requests_by_kind"][kind]
		if (
			actual["requests"] != expected["requests"]
			or actual["changes_per_request"] != [expected["changes_per_request"]] * expected["requests"]
		):
			raise ValueError(f"{kind} updates differ from the frozen contract: {trial_id}")
	if counts["query_requests"] != contract["query_requests"] or counts["query_rows"] != contract["query_rows"]:
		raise ValueError(f"Query workload differs from the frozen contract: {trial_id}")


def run(plan_path: Path, output: Path) -> dict:
	plan_path = plan_path.resolve()
	output = output.resolve()
	plan = json.loads(plan_path.read_text())
	trials = _validate_plan(plan)
	if output.exists():
		raise FileExistsError(f"External cohort output already exists: {output}")
	output.mkdir(parents=True)
	_save(output / "plan.json", plan)
	started = time.time()
	rows: list[dict] = []
	_save(output / "summary.json", {"plan_id": plan["id"], "state": "running", "trials": rows})
	for index, trial in enumerate(trials, 1):
		workload_path = Path(trial["workload"])
		checkpoint_value = trial.get("checkpoint", plan.get("checkpoint") if trial["dataset"] == "arxiv" else None)
		checkpoint = Path(checkpoint_value) if checkpoint_value else None
		recording = RecordedWorkload.load(workload_path, checkpoint=checkpoint)
		if recording.digest != trial["trace_digest"]:
			raise ValueError(f"Frozen workload digest changed: {trial['id']}")
		if recording.repetition != trial["repetition"]:
			raise ValueError(f"Workload repetition disagrees with plan: {trial['id']}")
		if recording.seed != trial.get("trace_seed"):
			raise ValueError(f"Workload trace seed disagrees with plan: {trial['id']}")
		_validate_workload_contract(recording.counts["measured"], plan["workload_contract"], trial["id"])
		trial_output = output / "trials" / trial["id"]
		phase = output / "phase.json"
		_save(
			phase,
			{
				"phase": "external_trial",
				"method": plan["display_name"],
				"trial": trial["id"],
				"trial_index": index,
				"trial_count": len(trials),
			},
		)
		try:
			report = run_external(
				plan["artifact"],
				Path(plan["source_checkout"]),
				trial_output,
				python_executable=Path(plan["python_executable"]),
				device="cpu",
				hardware_profile=plan["hardware_profile"],
				timeout=plan["trial_timeout_seconds"],
				seed=recording.seed,
				mode="replay",
				replay_options={
					"workload": workload_path,
					"graph": Path(trial["graph"]),
					"checkpoint": checkpoint,
					"processor": plan["processor"],
					"gcmean_compatibility": True,
					"add_inverse_edges": trial["dataset"] == "products",
				},
			)
			external_report = json.loads((trial_output / "verification.json").read_text())
			status = report["status"]
			error = report.get("failure")
			returncode = report.get("returncode")
			timed_out = status == "timeout"
		except Exception as exception:
			status = "harness_failed"
			error = f"{type(exception).__name__}: {exception}"
			returncode = None
			timed_out = False
			external_report = None
			trial_output.mkdir(parents=True, exist_ok=True)
			_save(trial_output / "harness-failure.json", {"status": status, "failure": error})
		_save(
			trial_output / "process-status.json",
			{"status": status, "returncode": returncode, "timed_out": timed_out, "failure": error},
		)
		row = {
			"id": trial["id"],
			"dataset": trial["dataset"],
			"repetition": trial["repetition"],
			"trace_digest": recording.digest,
			"input_identity": recording.input_identity,
			"status": status,
			"returncode": returncode,
			"timed_out": timed_out,
			"failure": error,
			"report": str((trial_output / "verification.json").relative_to(output))
			if external_report is not None
			else None,
			"timing": external_report.get("timing") if external_report else None,
			"correctness": external_report.get("correctness") if external_report else None,
			"oracle_scope": external_report.get("replay", {}).get("oracle_validation") if external_report else None,
		}
		rows.append(row)
		_save(
			output / "summary.json",
			{
				"plan_id": plan["id"],
				"cohort_kind": plan.get("cohort_kind", "planned"),
				"display_name": plan["display_name"],
				"state": "running",
				"started_unix": started,
				"completed_trials": len(rows),
				"trials": rows,
			},
		)
		print(json.dumps({"finished": trial["id"], "status": status}), flush=True)
	passed = sum(row["status"] == "passed" for row in rows)
	final = {
		"plan_id": plan["id"],
		"cohort_kind": plan.get("cohort_kind", "planned"),
		"display_name": plan["display_name"],
		"state": "completed",
		"started_unix": started,
		"finished_unix": time.time(),
		"completed_trials": len(rows),
		"passed_trials": passed,
		"valid_performance_trials": sum(
			row["timing"] is not None and row["timing"].get("metrics_valid") is True for row in rows
		),
		"trials": rows,
	}
	_save(output / "summary.json", final)
	_save(output / "phase.json", {"phase": "complete", "method": plan["display_name"], "completed": len(rows)})
	return final


app = typer.Typer(help=__doc__)


@app.command()
def main(
	plan: Annotated[Path, typer.Option("--plan")],
	output: Annotated[Path, typer.Option("--output")],
) -> None:
	result = run(plan, output)
	print(
		json.dumps(
			{key: result[key] for key in ("plan_id", "completed_trials", "passed_trials", "valid_performance_trials")}
		)
	)
	if result["passed_trials"] != result["completed_trials"]:
		raise SystemExit(1)


if __name__ == "__main__":
	app()
