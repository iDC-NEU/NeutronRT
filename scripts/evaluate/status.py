"""Create an immutable status manifest for completed and incomplete experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer


def summarize(path: Path) -> dict:
	summary_path = path / "summary.json"
	if not summary_path.is_file():
		return {"directory": str(path), "state": "missing-summary", "trials": []}
	summary = json.loads(summary_path.read_text())
	trials = []
	for item in summary.get("trials", []):
		trial_path = path / item["file"]
		if not trial_path.is_file():
			trials.append({**item, "state": "missing-result", "valid": False})
			continue
		trial = json.loads(trial_path.read_text())
		profile = trial.get("profile") or {}
		publication = profile.get("publication") or {}
		counts = trial.get("workload_counts") or {}
		effective = counts.get("effective_edge_changes") or {}
		network = (profile.get("network") or {}).get("totals") or {}
		trials.append(
			{
				**item,
				"state": (
					"complete" if trial.get("valid") and publication.get("service_drained", True) else "incomplete"
				),
				"valid": bool(trial.get("valid")),
				"trace_digest": trial.get("trace_digest"),
				"requested_input_changes": counts.get("measured", {}).get("input_changes"),
				"effective_edge_delta": effective.get("total_effective_edge_delta"),
				"effective_edge_changes": effective.get("total_effective_edge_changes"),
				"network_bytes_sent": network.get("bytes_sent"),
				"network_bytes_recv": network.get("bytes_recv"),
				"drained": publication.get("service_drained"),
			}
		)
	return {
		"directory": str(path),
		"state": "complete" if trials and all(item["state"] == "complete" for item in trials) else "incomplete",
		"method_order": [item["method"] for item in trials],
		"trials": trials,
	}


def markdown(reports: list[dict]) -> str:
	lines = [
		"Experiment status. Existing result directories are read-only inputs.",
		"",
		"| Directory | State | Complete | Incomplete |",
		"| --- | --- | ---: | ---: |",
	]
	for report in reports:
		complete = sum(item["state"] == "complete" for item in report["trials"])
		lines.append(
			f"| `{report['directory']}` | {report['state']} | {complete} | {len(report['trials']) - complete} |"
		)
	return "\n".join(lines) + "\n"


app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.command()
def main(
	inputs: Annotated[list[Path], typer.Argument(help="Result directories containing summary.json")],
	output: Annotated[Path, typer.Option("--output", help="New directory; existing output is never overwritten")],
):
	if output.exists():
		raise typer.BadParameter("Choose a new output directory to preserve previous status manifests")
	reports = [summarize(path) for path in inputs]
	output.mkdir(parents=True)
	(output / "status.json").write_text(json.dumps({"reports": reports}, indent=2) + "\n")
	(output / "status.md").write_text(markdown(reports))
	typer.echo(str(output))


if __name__ == "__main__":
	app()
