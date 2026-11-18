"""Load an already downloaded OGB dataset without interactive updates or downloads."""

import fcntl
import os
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from ogb.nodeproppred import DglNodePropPredDataset


@contextmanager
def _dataset_lock(root: Path):
	"""Serialize archive extraction so only one complete tree can be published."""
	lock_path = root / ".papers100M.lock"
	with lock_path.open("a") as lock:
		fcntl.flock(lock, fcntl.LOCK_EX)
		try:
			yield
		finally:
			fcntl.flock(lock, fcntl.LOCK_UN)


def _validate_download_name(value: object) -> str:
	if not isinstance(value, str):
		raise ValueError("OGB metadata has no valid archive directory name")
	path = Path(value)
	if not value or path.is_absolute() or path.parts != (value,):
		raise ValueError(f"Refusing unsafe OGB archive directory name: {value!r}")
	return value


def _extract_archive(archive: Path, staging: Path) -> None:
	root = staging.resolve()
	seen: set[str] = set()
	with zipfile.ZipFile(archive) as stream:
		for member in stream.infolist():
			if member.filename in seen:
				raise ValueError(f"Duplicate path in Papers100M archive: {member.filename}")
			seen.add(member.filename)
			member_path = Path(member.filename)
			if not member.filename or member_path.is_absolute() or ".." in member_path.parts:
				raise ValueError("Unsafe path in Papers100M archive")
			if (member.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
				raise ValueError("Symlinks are not allowed in Papers100M archive")
			if not (staging / member_path).resolve().is_relative_to(root):
				raise ValueError("Unsafe path in Papers100M archive")
		stream.extractall(staging)


def _validate_offline_path(path: Path, metadata: dict, *, require_raw: bool) -> None:
	release = path / ("RELEASE_v" + str(metadata["version"]) + ".txt")
	if not release.is_file():
		raise FileNotFoundError(f"Offline preparation requires the archive's release marker: {release}")
	if require_raw or not (path / "processed" / "dgl_data_processed").is_file():
		for filename in ("data.npz", "node-label.npz"):
			if not (path / "raw" / filename).is_file():
				raise FileNotFoundError(f"Offline preparation requires downloaded raw file: {path / 'raw' / filename}")


def papers100m(root: Path):
	"""Use the official reader only after checking its offline prerequisites.

	The existing download task owns transfer/extraction. Missing release or raw
	files must fail preparation rather than prompting, deleting or redownloading.
	"""
	import ogb.nodeproppred

	name = "ogbn-papers100M"
	master = Path(ogb.nodeproppred.__file__).parent / "master.csv"
	metadata = pd.read_csv(master, index_col=0, keep_default_na=False)[name].to_dict()
	_download_name = _validate_download_name(metadata.get("download_name"))
	root.mkdir(parents=True, exist_ok=True)
	path = root / "ogbn_papers100M"
	if path.with_name(path.name + "_dgl").exists():
		path = path.with_name(path.name + "_dgl")
	archive = root / "papers100M-bin.zip"
	if not path.exists() and archive.is_file():
		with _dataset_lock(root):
			if not path.exists():
				with tempfile.TemporaryDirectory(prefix=".papers-unpack-", dir=root) as folder:
					staging = Path(folder)
					_extract_archive(archive, staging)
					extracted = staging / _download_name
					if not extracted.is_dir():
						raise ValueError("Archive does not contain expected OGB dataset directory")
					_validate_offline_path(extracted, metadata, require_raw=True)
					os.rename(extracted, path)
	_validate_offline_path(path, metadata, require_raw=False)
	metadata["dir_path"] = str(path)
	return DglNodePropPredDataset(name=name, meta_dict=metadata)
