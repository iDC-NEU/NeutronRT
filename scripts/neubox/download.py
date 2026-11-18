"""Resume an archive with bounded parallel HTTP connections; run inside neu-sbox."""

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import typer


@contextmanager
def _download_lock(target: Path):
	"""Serialize writers for one target while keeping the lock across failures."""
	lock_path = target.with_name(target.name + ".lock")
	with lock_path.open("a") as lock:
		fcntl.flock(lock, fcntl.LOCK_EX)
		try:
			yield
		finally:
			fcntl.flock(lock, fcntl.LOCK_UN)


def _write_resume_identity(path: Path, url: str) -> None:
	identity = {"url_sha256": hashlib.sha256(url.encode()).hexdigest()}
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
		) as stream:
			temporary = Path(stream.name)
			json.dump(identity, stream)
			stream.write("\n")
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temporary, path)
	finally:
		if temporary is not None and temporary.exists():
			temporary.unlink()


def _prepare_resume(partial: Path, identity_path: Path, url: str) -> None:
	if partial.exists() and not partial.is_file():
		raise ValueError(f"Resume path is not a regular file: {partial}")
	if partial.exists():
		if not identity_path.is_file():
			raise ValueError(f"Refusing to resume an unidentified partial archive: {partial}")
		try:
			identity = json.loads(identity_path.read_text(encoding="utf-8"))
		except (OSError, json.JSONDecodeError) as error:
			raise ValueError(f"Resume metadata is invalid: {identity_path}") from error
		expected = hashlib.sha256(url.encode()).hexdigest()
		if identity.get("url_sha256") != expected:
			raise ValueError("Partial archive belongs to a different URL; refusing to resume it")
	elif identity_path.exists():
		identity_path.unlink()
	_write_resume_identity(identity_path, url)


def _publish(partial: Path, target: Path, identity_path: Path) -> None:
	"""Publish a complete file without replacing a target created by another writer."""
	try:
		os.link(partial, target)
	except FileExistsError as error:
		raise FileExistsError(f"Completed target appeared while downloading: {target}") from error
	partial.unlink()
	identity_path.unlink(missing_ok=True)
	try:
		directory = os.open(target.parent, os.O_RDONLY)
		try:
			os.fsync(directory)
		finally:
			os.close(directory)
	except OSError:
		# The target is already atomically visible; directory fsync is best effort.
		pass


def main(
	url: str,
	target: Path,
	connections: int = 8,
	proxy: str | None = None,
	attempts: int = 3,
	seconds: int = 43200,
):
	if not 1 <= connections <= 16:
		raise typer.BadParameter("Use 1–16 connections")
	if not 1 <= attempts <= 8 or not 1 <= seconds <= 43200:
		raise typer.BadParameter("Use 1–8 attempts within a total download budget of 1–43200 seconds")
	target.parent.mkdir(parents=True, exist_ok=True)
	with _download_lock(target):
		if target.exists():
			raise FileExistsError("Completed target exists; refusing to overwrite it")
		partial = target.with_name(target.name + ".part")
		identity_path = target.with_name(target.name + ".part.json")
		_prepare_resume(partial, identity_path, url)
		args = [
			"timeout",
			"--foreground",
			"--kill-after=30s",
			f"{seconds}s",
			"aria2c",
			"--continue=true",
			"--always-resume=true",
			"--allow-overwrite=false",
			"--auto-file-renaming=false",
			"--file-allocation=none",
			f"--max-connection-per-server={connections}",
			f"--split={connections}",
			"--min-split-size=16M",
			"--connect-timeout=15",
			"--timeout=30",
			"--max-tries=8",
			"--retry-wait=10",
			"--summary-interval=30",
			"--download-result=full",
			"--console-log-level=notice",
			"--dir",
			str(target.parent),
			"--out",
			partial.name,
		]
		if proxy:
			args += ["--all-proxy", proxy]
		deadline = time.monotonic() + seconds
		for attempt in range(1, attempts + 1):
			remaining = int(deadline - time.monotonic())
			if remaining < 1:
				raise typer.Exit(124)
			args[3] = f"{remaining}s"
			typer.echo(f"Download attempt {attempt}/{attempts}; resuming preserved partial, budget={remaining}s")
			result = subprocess.run([*args, url], check=False)
			if result.returncode == 0:
				break
			# Retry aria2 EOF (1) and network timeout (2), preserving its block
			# map. GNU timeout (124), interrupts and other fatal exits stop.
			if result.returncode not in (1, 2) or attempt == attempts:
				raise typer.Exit(result.returncode)
			remaining = int(deadline - time.monotonic())
			if remaining <= 1:
				raise typer.Exit(124)
			time.sleep(min(10.0, remaining - 1))
		if not partial.is_file():
			raise FileNotFoundError(f"Downloader reported success but produced no archive: {partial}")
		with zipfile.ZipFile(partial) as archive:
			bad = archive.testzip()
			if bad:
				raise ValueError(f"Archive CRC failed for {bad}; partial file preserved")
		# Publish only after the downloader and ZIP CRC validation both succeeded.
		_publish(partial, target, identity_path)


if __name__ == "__main__":
	typer.run(main)
