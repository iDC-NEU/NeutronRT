"""Baseline provenance must reject changed code and exclude local artifacts."""

import subprocess

import pytest

from benchmarks.sources import PinnedSource


@pytest.fixture
def pinned(tmp_path):
	checkout = tmp_path / "checkout"
	checkout.mkdir()

	def git(*args):
		return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.PIPE).decode().strip()

	git("init")
	(checkout / "algorithm.py").write_text("VALUE = 1\n")
	(checkout / ".gitignore").write_text("private/\n")
	git("add", ".")
	git("-c", "user.name=Baseline test", "-c", "user.email=baseline@example.invalid", "commit", "-m", "test fixture")
	pin = PinnedSource("test", git("rev-parse", "HEAD"), "https://example.invalid/baseline")
	return checkout, pin


def test_snapshot_cannot_import_ignored_local_code_or_include_git_history(pinned, tmp_path):
	checkout, pin = pinned
	(checkout / "private").mkdir()
	(checkout / "private/shadow.py").write_text("raise RuntimeError('local artifact')\n")
	snapshot = pin.snapshot(checkout, tmp_path / "snapshot")
	assert (snapshot / "algorithm.py").read_bytes() == (checkout / "algorithm.py").read_bytes()
	assert not (snapshot / ".git").exists()
	assert not (snapshot / "private").exists()


@pytest.mark.parametrize("change", ["tracked", "untracked", "revision"])
def test_changed_baseline_is_rejected_before_export(pinned, tmp_path, change):
	checkout, pin = pinned
	if change == "tracked":
		(checkout / "algorithm.py").write_text("VALUE = 2\n")
	elif change == "untracked":
		(checkout / "injected.py").write_text("VALUE = 2\n")
	else:
		pin = PinnedSource(pin.name, "0" * 40, pin.repository)
	output = tmp_path / "snapshot"
	with pytest.raises(ValueError):
		pin.snapshot(checkout, output)
	assert not output.exists()
