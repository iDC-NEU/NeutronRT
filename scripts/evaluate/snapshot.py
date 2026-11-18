"""Create or verify a frozen source tree for scheduler submissions."""

import json
from pathlib import Path

from lib.profiling.snapshot import digest as digest
from lib.profiling.snapshot import freeze as freeze
from lib.profiling.snapshot import verify as verify


def main(root: Path, output: Path | None = None):
	"""Verify a frozen source tree, or export a new one with --output."""
	metadata = freeze(root.resolve(), output.resolve()) if output else verify(root.resolve())
	print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
	import typer

	typer.run(main)
