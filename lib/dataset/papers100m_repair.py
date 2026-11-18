"""Repair corrupted members of a complete Papers100M ZIP in place.

The public archive is large enough that restarting a mostly complete transfer
is undesirable. This utility verifies every member, re-downloads only the
compressed byte ranges whose CRC or decompressor check fails, and publishes
the repaired archive only after a second complete validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import time
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from http.client import HTTPException
from pathlib import Path

DEFAULT_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
_LOCAL_HEADER = b"PK\x03\x04"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")
_SCAN_CHUNK_BYTES = 16 * 1024 * 1024
_REPAIR_WINDOW_BYTES = 64 * 1024 * 1024
_RANGE_CHUNK_BYTES = 32 * 1024 * 1024
_RANGE_ATTEMPTS = 8


@dataclass(frozen=True)
class BadMember:
	name: str
	start: int
	length: int
	error: str


def _url_sha256(url: str) -> str:
	return hashlib.sha256(url.encode()).hexdigest()


def _check_resume_identity(source: Path, url: str) -> None:
	identity = source.with_name(source.name + ".json")
	if not identity.exists():
		return
	try:
		value = json.loads(identity.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError) as error:
		raise ValueError(f"Invalid resume metadata: {identity}") from error
	if value.get("url_sha256") != _url_sha256(url):
		raise ValueError(f"Resume metadata belongs to a different URL: {identity}")


def _member_data_range(stream, info: zipfile.ZipInfo) -> tuple[int, int]:
	stream.seek(info.header_offset)
	header = stream.read(30)
	if len(header) != 30 or header[:4] != _LOCAL_HEADER:
		raise zipfile.BadZipFile(f"Invalid local header for {info.filename}")
	name_length, extra_length = struct.unpack_from("<HH", header, 26)
	return info.header_offset + 30 + name_length + extra_length, info.compress_size


def _scan_deflated_member(stream, info: zipfile.ZipInfo, *, report) -> BadMember | None:
	start, length = _member_data_range(stream, info)
	end = start + length
	decompressor = zlib.decompressobj(-15)
	checksum = 0
	file_size = 0
	position = start
	last_report = start
	stream.seek(start)
	while position < end:
		chunk = stream.read(min(_SCAN_CHUNK_BYTES, end - position))
		if not chunk:
			window_start = max(start, position - _REPAIR_WINDOW_BYTES // 2)
			return BadMember(
				info.filename,
				window_start,
				min(_REPAIR_WINDOW_BYTES, end - window_start),
				"unexpected end of compressed member",
			)
		try:
			output = decompressor.decompress(chunk)
		except zlib.error as error:
			window_start = max(start, position - _REPAIR_WINDOW_BYTES // 2)
			window_end = min(end, position + len(chunk) + _REPAIR_WINDOW_BYTES // 2)
			return BadMember(info.filename, window_start, window_end - window_start, f"{type(error).__name__}: {error}")
		checksum = zlib.crc32(output, checksum)
		file_size += len(output)
		position += len(chunk)
		if position - last_report >= 1024 * 1024 * 1024:
			report(f"Scanning {info.filename}: {(position - start) / 1024**3:.1f} GiB / {length / 1024**3:.1f} GiB")
			last_report = position
	if not decompressor.eof or decompressor.unused_data:
		window_start = max(start, end - _REPAIR_WINDOW_BYTES)
		return BadMember(info.filename, window_start, end - window_start, "compressed member did not end cleanly")
	try:
		decompressor.flush()
	except zlib.error as error:
		window_start = max(start, end - _REPAIR_WINDOW_BYTES)
		return BadMember(info.filename, window_start, end - window_start, f"{type(error).__name__}: {error}")
	if file_size != info.file_size or checksum & 0xFFFFFFFF != info.CRC:
		return BadMember(info.filename, start, length, "member size or CRC does not match ZIP metadata")
	return None


def _scan_bad_members(source: Path, *, report=print) -> list[BadMember]:
	bad: list[BadMember] = []
	with zipfile.ZipFile(source) as archive, source.open("rb") as stream:
		for info in archive.infolist():
			if info.is_dir():
				continue
			report(f"Validating {info.filename} ({info.compress_size / 1024**2:.1f} MiB)")
			try:
				if info.compress_type == zipfile.ZIP_DEFLATED:
					failure = _scan_deflated_member(stream, info, report=report)
				else:
					with archive.open(info) as member:
						while member.read(1024 * 1024):
							pass
					failure = None
			except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
				start, length = _member_data_range(stream, info)
				failure = BadMember(info.filename, start, length, f"{type(error).__name__}: {error}")
			if failure is not None:
				bad.append(failure)
				report(
					f"Corruption found in {failure.name}: {failure.error}; "
					f"repair range {failure.start}:{failure.length}"
				)
	return bad


def _range_opener(proxy: str | None):
	if proxy is None:
		return urllib.request.build_opener()
	return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))


def _download_range_once(url: str, start: int, length: int, *, proxy: str | None, timeout: float, output: Path) -> None:
	if length < 0:
		raise ValueError(f"Negative range length: {length}")
	end = start + length - 1
	request = urllib.request.Request(
		url,
		headers={"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"},
	)
	opener = _range_opener(proxy)
	with opener.open(request, timeout=timeout) as response:
		status = getattr(response, "status", response.getcode())
		content_range = response.headers.get("Content-Range", "")
		match = _CONTENT_RANGE.fullmatch(content_range)
		if status != 206 or match is None or int(match.group(1)) != start or int(match.group(2)) != end:
			raise OSError(f"Server did not honor byte range {start}-{end}: status={status}, range={content_range!r}")
		remaining = length
		with output.open("wb") as stream:
			while remaining:
				chunk = response.read(min(1024 * 1024, remaining))
				if not chunk:
					raise OSError(f"Range ended early at {start}; {remaining} bytes remain")
				stream.write(chunk)
				remaining -= len(chunk)
			stream.flush()
			os.fsync(stream.fileno())
	if output.stat().st_size != length:
		raise OSError(f"Downloaded range has the wrong size: expected {length}, found {output.stat().st_size}")


def _download_range(
	url: str,
	start: int,
	length: int,
	*,
	proxy: str | None,
	timeout: float,
	output: Path,
	report,
) -> None:
	"""Fetch a range in small independently retriable requests."""
	if length < 0:
		raise ValueError(f"Negative range length: {length}")
	output.unlink(missing_ok=True)
	with output.open("wb"):
		pass
	for offset in range(0, length, _RANGE_CHUNK_BYTES):
		chunk_length = min(_RANGE_CHUNK_BYTES, length - offset)
		for attempt in range(1, _RANGE_ATTEMPTS + 1):
			with tempfile.NamedTemporaryFile(
				prefix=f".{output.name}.", suffix=".chunk", dir=output.parent, delete=False
			) as stream:
				chunk_path = Path(stream.name)
			try:
				report(
					f"Downloading repair range {start + offset}:{chunk_length} "
					f"({offset / 1024**2:.0f}/{length / 1024**2:.0f} MiB), attempt {attempt}"
				)
				_download_range_once(
					url,
					start + offset,
					chunk_length,
					proxy=proxy,
					timeout=timeout,
					output=chunk_path,
				)
				with output.open("ab") as destination, chunk_path.open("rb") as source:
					while chunk := source.read(1024 * 1024):
						destination.write(chunk)
					destination.flush()
					os.fsync(destination.fileno())
				break
			except (OSError, TimeoutError, HTTPException) as error:
				if attempt == _RANGE_ATTEMPTS:
					raise
				report(f"Range attempt failed: {error}; retrying")
				time.sleep(min(30, 2**attempt))
			finally:
				chunk_path.unlink(missing_ok=True)
		else:
			raise AssertionError("unreachable range retry state")
	if output.stat().st_size != length:
		raise OSError(f"Downloaded range has the wrong size: expected {length}, found {output.stat().st_size}")


def _replace_range(source: Path, start: int, replacement: Path) -> None:
	length = replacement.stat().st_size
	with source.open("r+b") as stream, replacement.open("rb") as patch:
		stream.seek(start)
		remaining = length
		while remaining:
			chunk = patch.read(min(1024 * 1024, remaining))
			if not chunk:
				raise OSError("Replacement range ended early")
			stream.write(chunk)
			remaining -= len(chunk)
		stream.flush()
		os.fsync(stream.fileno())


def _sha256(source: Path) -> str:
	digest = hashlib.sha256()
	with source.open("rb") as stream:
		while chunk := stream.read(1024 * 1024):
			digest.update(chunk)
	return digest.hexdigest()


def repair(
	source: Path,
	target: Path,
	*,
	url: str = DEFAULT_URL,
	proxy: str | None = None,
	timeout: float = 60,
) -> dict:
	"""Repair ``source`` and publish it as ``target`` after full validation."""
	if not source.is_file():
		raise FileNotFoundError(source)
	if target.exists():
		raise FileExistsError(f"Refusing to overwrite completed archive: {target}")
	_check_resume_identity(source, url)

	def report(message: str) -> None:
		print(message, flush=True)

	report("Scanning archive for corrupt members")
	bad = _scan_bad_members(source, report=report)
	for index, member in enumerate(bad):
		if member.length == 0:
			raise zipfile.BadZipFile(f"Corrupt empty member cannot be repaired by range download: {member.name}")
		with tempfile.NamedTemporaryFile(
			prefix=f".{source.name}.{index:04d}.", suffix=".range", dir=source.parent, delete=False
		) as stream:
			range_path = Path(stream.name)
		try:
			_download_range(
				url,
				member.start,
				member.length,
				proxy=proxy,
				timeout=timeout,
				output=range_path,
				report=report,
			)
			_replace_range(source, member.start, range_path)
		finally:
			range_path.unlink(missing_ok=True)
	report("Revalidating repaired archive")
	remaining = _scan_bad_members(source, report=report)
	if remaining:
		raise zipfile.BadZipFile(f"Archive remains corrupt after repair: {[item.name for item in remaining]}")
	target.parent.mkdir(parents=True, exist_ok=True)
	try:
		os.link(source, target)
	except FileExistsError as error:
		raise FileExistsError(f"Completed archive appeared while repairing: {target}") from error
	identity = source.with_name(source.name + ".json")
	identity.unlink(missing_ok=True)
	source.unlink()
	return {
		"archive": str(target),
		"bytes": target.stat().st_size,
		"sha256": _sha256(target),
		"repaired_members": [asdict(item) for item in bad],
		"status": "ready",
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("source", type=Path)
	parser.add_argument("target", type=Path)
	parser.add_argument("--url", default=DEFAULT_URL)
	parser.add_argument("--proxy")
	parser.add_argument("--timeout", type=float, default=60)
	args = parser.parse_args()
	print(json.dumps(repair(args.source, args.target, url=args.url, proxy=args.proxy, timeout=args.timeout), indent=2))


if __name__ == "__main__":
	main()
