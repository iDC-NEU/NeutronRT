"""Check released InkStream SAGE in the documented isolated dependency environment."""

import json
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.inkstream import verify


def main(
	source: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Clean checkout at the pinned release")],
	output: Annotated[Path, typer.Option(help="New directory for source snapshot and verification report")],
	seed: Annotated[int, typer.Option(min=0)] = 71,
) -> None:
	report = verify(source.resolve(), output.resolve(), seed=seed)
	print(
		json.dumps(
			{"valid": report["valid"], "checks": len(report["records"]), "revision": report["upstream_revision"]}
		)
	)
	if not report["valid"]:
		raise typer.Exit(1)


if __name__ == "__main__":
	typer.run(main)
