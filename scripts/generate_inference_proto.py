"""Regenerate the owned RPC stubs with the generator pinned in requirements.txt."""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import typer


def main():
	if version("grpcio-tools") != "1.75.1":
		raise RuntimeError("Use grpcio-tools==1.75.1 from requirements.txt to regenerate these stubs")
	root = Path(__file__).resolve().parents[1]
	subprocess.run(
		[
			sys.executable,
			"-m",
			"grpc_tools.protoc",
			"-I.",
			"--python_out=.",
			"--pyi_out=.",
			"--grpc_python_out=.",
			"lib/rpc/inference.proto",
		],
		cwd=root,
		check=True,
	)
	for name, rule in (("inference_pb2.py", "E501"), ("inference_pb2_grpc.py", "E101")):
		path = root / "lib" / "rpc" / name
		path.write_text(f"# ruff: noqa: {rule}\n" + path.read_text())
	paths = [
		"lib/rpc/inference_pb2.py",
		"lib/rpc/inference_pb2.pyi",
		"lib/rpc/inference_pb2_grpc.py",
	]
	subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "--ignore", "E501", *paths], cwd=root, check=True)
	subprocess.run([sys.executable, "-m", "ruff", "format", *paths], cwd=root, check=True)
	subprocess.run([sys.executable, "-m", "ruff", "check", *paths], cwd=root, check=True)


if __name__ == "__main__":
	typer.run(main)
