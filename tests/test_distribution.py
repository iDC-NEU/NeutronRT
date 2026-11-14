"""Native/IDL resources must survive a normal wheel build without downloads."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_wheel_includes_the_runtime_native_source_and_rpc_schema(tmp_path):
	root = Path(__file__).resolve().parents[1]
	source = tmp_path / "source"
	source.mkdir()
	shutil.copy2(root / "pyproject.toml", source / "pyproject.toml")
	shutil.copytree(root / "lib", source / "lib", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
	destination = tmp_path / "wheels"
	subprocess.run(
		[
			sys.executable,
			"-m",
			"pip",
			"wheel",
			"--no-deps",
			"--no-build-isolation",
			"--no-index",
			"--wheel-dir",
			str(destination),
			str(source),
		],
		check=True,
		capture_output=True,
		text=True,
		env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
		timeout=60,
	)
	with ZipFile(next(destination.glob("*.whl"))) as wheel:
		for path in ("lib/graph/csrc/parallel.cpp", "lib/rpc/inference.proto"):
			assert wheel.read(path) == (root / path).read_bytes()
