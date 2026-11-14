"""Verify frozen ICDE baseline kernels using their original implementations."""

import json
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.icde import verify


def main(
	source: Annotated[
		Path, typer.Option(exists=True, file_okay=False, help="Clean checkout at the frozen ICDE revision")
	],
	output: Annotated[Path, typer.Option(help="New directory for source snapshot and verification report")],
	fanout: Annotated[list[int] | None, typer.Option(min=1, help="Repeat for each fanout; default 1, 5, 10")] = None,
	seed: Annotated[int, typer.Option(min=0)] = 71,
	sampling_seeds: Annotated[int, typer.Option(min=1)] = 8,
) -> None:
	report = verify(
		source.resolve(), output.resolve(), fanouts=fanout or [1, 5, 10], seed=seed, sampling_seeds=sampling_seeds
	)
	print(
		json.dumps(
			{"valid": report["valid"], "checks": len(report["records"]), "revision": report["upstream_revision"]}
		)
	)
	if not report["valid"]:
		raise typer.Exit(1)


if __name__ == "__main__":
	typer.run(main)
