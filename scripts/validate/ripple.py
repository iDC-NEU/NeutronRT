"""Check the pinned authors' Ripple processors against their full DGL model."""

import json
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.ripple import verify


def main(
	source: Annotated[
		Path, typer.Option(exists=True, file_okay=False, help="Unchanged checkout at the pinned revision")
	],
	output: Annotated[
		Path, typer.Option(help="New directory for source snapshot, generated fixtures and verification")
	],
	seed: Annotated[int, typer.Option(min=0)] = 71,
) -> None:
	report = verify(source.resolve(), output.resolve(), seed=seed)
	print(
		json.dumps(
			{"valid": report["valid"], "checks": len(report["records"]), "revision": report["upstream_revision"]}
		)
	)


if __name__ == "__main__":
	typer.run(main)
