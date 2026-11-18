"""Map the current Neu Box GPU allocation to one stable CUDA-visible UUID."""

import os
import sys
from typing import Annotated

from scripts.peer.allocation import AllocationMetadataError, resolve_current_allocation


def main(command: list[str] | None = None) -> None:
	arguments = sys.argv[1:] if command is None else command
	if arguments and arguments[0] == "--":
		arguments = arguments[1:]
	if not arguments:
		raise SystemExit("usage: python -m scripts.peer.gpu_exec -- COMMAND [ARG ...]")
	allocation, gpu_uuids = resolve_current_allocation()
	if allocation is None:
		raise AllocationMetadataError("GPU worker is not running inside a Neu Box allocation")
	if len(gpu_uuids) != 1:
		raise AllocationMetadataError(f"GPU peer worker requires exactly one allocated GPU, found {len(gpu_uuids)}")
	environment = os.environ.copy()
	environment["CUDA_VISIBLE_DEVICES"] = gpu_uuids[0]
	os.execvpe(arguments[0], arguments, environment)


def create_app():
	"""Build a variadic Typer entrypoint while keeping the bootstrap import-light."""
	import typer

	app = typer.Typer(
		no_args_is_help=True,
		context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
	)

	@app.command()
	def execute(command: Annotated[list[str], typer.Argument(help="Command and arguments to exec")]) -> None:
		main(command)

	return app


def cli() -> None:
	create_app()()


if __name__ == "__main__":
	cli()
