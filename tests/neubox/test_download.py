import json
import zipfile
from types import SimpleNamespace

import pytest

from scripts.neubox.download import main


def _write_zip(path):
	with zipfile.ZipFile(path, "w") as archive:
		archive.writestr("fixture.txt", "complete")


def test_download_publishes_only_after_crc_and_cleans_resume_metadata(monkeypatch, tmp_path):
	target = tmp_path / "papers100M-bin.zip"
	url = "https://example.invalid/papers100M-bin.zip"

	def run(args, **kwargs):
		assert "--always-resume=true" in args
		assert "--allow-overwrite=false" in args
		_write_zip(target.with_name(target.name + ".part"))
		return SimpleNamespace(returncode=0)

	monkeypatch.setattr("scripts.neubox.download.subprocess.run", run)
	main(url, target, connections=2)

	assert zipfile.ZipFile(target).read("fixture.txt") == b"complete"
	assert not target.with_name(target.name + ".part").exists()
	assert not target.with_name(target.name + ".part.json").exists()


def test_download_refuses_to_resume_partial_from_another_url(monkeypatch, tmp_path):
	target = tmp_path / "archive.zip"
	partial = target.with_name(target.name + ".part")
	partial.write_bytes(b"partial")
	partial.with_name(partial.name + ".json").write_text(json.dumps({"url_sha256": "wrong"}))
	monkeypatch.setattr("scripts.neubox.download.subprocess.run", lambda *args, **kwargs: None)

	with pytest.raises(ValueError, match="different URL"):
		main("https://example.invalid/new.zip", target)
	assert partial.read_bytes() == b"partial"


def test_download_does_not_replace_target_created_during_transfer(monkeypatch, tmp_path):
	target = tmp_path / "archive.zip"

	def run(args, **kwargs):
		_write_zip(target.with_name(target.name + ".part"))
		target.write_bytes(b"winner")
		return SimpleNamespace(returncode=0)

	monkeypatch.setattr("scripts.neubox.download.subprocess.run", run)
	with pytest.raises(FileExistsError, match="appeared"):
		main("https://example.invalid/archive.zip", target)
	assert target.read_bytes() == b"winner"


@pytest.mark.parametrize("transport_exit", [1, 2])
def test_transport_retry_preserves_partial_and_uses_remaining_total_budget(monkeypatch, tmp_path, transport_exit):
	target = tmp_path / "archive.zip"
	partial = target.with_name(target.name + ".part")
	calls = []
	clock = iter([0.0, 1.0, 20.0, 30.0])
	monkeypatch.setattr("scripts.neubox.download.time.monotonic", lambda: next(clock))
	monkeypatch.setattr("scripts.neubox.download.time.sleep", lambda _: None)

	def run(args, **kwargs):
		calls.append(args)
		if len(calls) == 1:
			partial.write_bytes(b"preserve")
			return SimpleNamespace(returncode=transport_exit)
		assert partial.read_bytes() == b"preserve"
		_write_zip(partial)
		return SimpleNamespace(returncode=0)

	monkeypatch.setattr("scripts.neubox.download.subprocess.run", run)
	main("https://example.invalid/archive.zip", target, attempts=2, seconds=100)
	assert [call[3] for call in calls] == ["99s", "70s"]
	assert target.is_file()
