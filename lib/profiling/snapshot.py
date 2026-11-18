"""Create and verify source snapshots for scheduler submissions."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

SOURCE_DIRS = {"lib", "scripts", "benchmarks", "config", "examples", "tests"}
SOURCE_FILES = {"pyproject.toml", "requirements.txt", "requirements-dev.txt"}


def digest(path: Path) -> str:
	with path.open("rb") as stream:
		return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(root: Path, destination: Path) -> dict:
	"""Copy tracked and untracked source, including local edits, without Git/private artifacts."""
	paths = (
		subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
		.decode()
		.split("\0")
	)
	metadata: dict[str, Any] = {
		"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
		"files": {},
	}
	destination.mkdir(parents=True, exist_ok=False)
	for relative in sorted(set(paths)):
		path = Path(relative)
		if not relative or path.parts[0] not in SOURCE_DIRS and relative not in SOURCE_FILES:
			continue
		source = root / path
		if not source.is_file():
			continue
		if not source.resolve().is_relative_to(root.resolve()):
			raise ValueError(f"Source symlink escapes repository: {relative}")
		target = destination / path
		target.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(source, target)
		metadata["files"][relative] = digest(target)
	metadata["tree_sha256"] = hashlib.sha256(json.dumps(metadata["files"], sort_keys=True).encode()).hexdigest()
	(destination / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root))
	(destination / "source-snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
	return metadata


def verify(root: Path) -> dict:
	metadata: dict[str, Any] = json.loads((root / "source-snapshot.json").read_text())
	actual = {
		path.relative_to(root).as_posix()
		for directory in SOURCE_DIRS
		for path in (root / directory).rglob("*")
		if path.is_file() and "__pycache__" not in path.relative_to(root).parts
	}
	actual.update(name for name in SOURCE_FILES if (root / name).is_file())
	if actual != set(metadata["files"]):
		raise ValueError("Frozen source file set changed")
	for relative, expected in metadata["files"].items():
		if digest(root / relative) != expected:
			raise ValueError(f"Frozen source changed: {relative}")
	return metadata
