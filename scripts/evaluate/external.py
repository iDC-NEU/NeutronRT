"""Run an independent pinned external baseline verification gate."""

import json
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.external import run_external

app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	artifact: Annotated[str, typer.Option("--artifact", help="Pinned artifact: ripple or inkstream")],
	source: Annotated[Path, typer.Option("--source", exists=True, file_okay=False)],
	output: Annotated[
		Path, typer.Option("--output", help="New output directory; existing results are never overwritten")
	],
	python_executable: Annotated[
		Path | None, typer.Option("--python", help="Python executable or PATH name for the isolated environment")
	] = None,
	device: Annotated[str, typer.Option("--device", help="cpu or an explicit cuda index")] = "cpu",
	hardware_profile: Annotated[
		str | None, typer.Option("--hardware-profile", help="Hardware profile path or descriptive label")
	] = None,
	timeout: Annotated[float, typer.Option("--timeout", min=0.1, max=3600)] = 300,
	seed: Annotated[int, typer.Option("--seed", min=0)] = 71,
	mode: Annotated[str, typer.Option("--mode", help="verify or gated Ripple replay")] = "verify",
	workload: Annotated[Path | None, typer.Option("--workload", exists=True, dir_okay=False)] = None,
	graph: Annotated[Path | None, typer.Option("--graph", exists=True)] = None,
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint", exists=True, dir_okay=False)] = None,
	processor: Annotated[str, typer.Option("--processor", help="Ripple processor: incremental or rc")] = "incremental",
	gcmean_compatibility: Annotated[bool, typer.Option("--gcmean-compatibility")] = False,
	add_inverse_edges: Annotated[bool, typer.Option("--add-inverse-edges")] = False,
	phase_file: Annotated[Path | None, typer.Option("--phase-file")] = None,
) -> None:
	if mode == "replay" and (workload is None or graph is None):
		raise typer.BadParameter("--mode replay requires --workload and --graph")
	if phase_file is not None:
		phase_file.parent.mkdir(parents=True, exist_ok=True)
		temporary = phase_file.with_suffix(phase_file.suffix + ".tmp")
		temporary.write_text(json.dumps({"phase": "external_replay", "method": artifact}) + "\n")
		temporary.replace(phase_file)
	report = run_external(
		artifact,
		source,
		output,
		python_executable=python_executable,
		device=device,
		hardware_profile=hardware_profile,
		timeout=timeout,
		seed=seed,
		mode=mode,
		replay_options={
			"workload": workload,
			"graph": graph,
			"checkpoint": checkpoint,
			"processor": processor if mode == "replay" else None,
			"gcmean_compatibility": gcmean_compatibility if mode == "replay" else None,
			"add_inverse_edges": add_inverse_edges if mode == "replay" else None,
		},
	)
	if phase_file is not None:
		temporary = phase_file.with_suffix(phase_file.suffix + ".tmp")
		temporary.write_text(
			json.dumps({"phase": "external_complete", "method": artifact, "status": report["status"]}) + "\n"
		)
		temporary.replace(phase_file)
	typer.echo(json.dumps(report, indent=2, sort_keys=True))
	if report["status"] != "passed":
		raise typer.Exit(1)


if __name__ == "__main__":
	app()
