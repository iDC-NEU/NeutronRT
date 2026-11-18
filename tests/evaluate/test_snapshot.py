import subprocess

import pytest

from scripts.evaluate.snapshot import freeze, verify


def test_snapshot_includes_dirty_and_untracked_code_and_detects_tampering(tmp_path):
	root = tmp_path / "repo"
	root.mkdir()
	subprocess.run(["git", "init", "-q", str(root)], check=True)
	(root / "lib").mkdir()
	(root / "lib/code.py").write_text("old = 1\n")
	subprocess.run(["git", "add", "."], cwd=root, check=True)
	subprocess.run(
		[
			"git",
			"-c",
			"user.name=Test",
			"-c",
			"user.email=test@example.invalid",
			"-c",
			"core.hooksPath=/dev/null",
			"commit",
			"-qm",
			"fixture",
		],
		cwd=root,
		check=True,
	)
	(root / "lib/code.py").write_text("dirty = 2\n")
	(root / "lib/new.py").write_text("new = 3\n")
	(root / "benchmarks").mkdir()
	(root / "benchmarks/sources.json").write_text("{}")
	(root / "paper").mkdir()
	(root / "paper/private.txt").write_text("excluded")
	destination = tmp_path / "snapshot"
	metadata = freeze(root, destination)
	(root / "lib/code.py").write_text("later = 4\n")
	assert (destination / "lib/code.py").read_text() == "dirty = 2\n"
	assert "lib/new.py" in metadata["files"]
	assert "benchmarks/sources.json" in metadata["files"]
	assert not (destination / "paper").exists()
	assert verify(destination) == metadata
	(destination / "lib/new.py").write_text("tampered\n")
	with pytest.raises(ValueError, match="Frozen source changed"):
		verify(destination)


def test_snapshot_rejects_new_importable_files_but_allows_bytecode_cache(tmp_path):
	import hashlib
	import json

	root = tmp_path / "snapshot"
	(root / "lib").mkdir(parents=True)
	(root / "lib/known.py").write_text("known = 1\n")
	metadata = {"files": {"lib/known.py": hashlib.sha256((root / "lib/known.py").read_bytes()).hexdigest()}}
	(root / "source-snapshot.json").write_text(json.dumps(metadata))
	(root / "lib/__pycache__").mkdir()
	(root / "lib/__pycache__/known.cpython.pyc").write_bytes(b"cache")
	assert verify(root) == metadata
	(root / "lib/injected.py").write_text("injected = True\n")
	with pytest.raises(ValueError, match="file set changed"):
		verify(root)
