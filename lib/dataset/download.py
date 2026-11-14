"""Resumable, atomic downloads of public graph archives."""

import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from pathlib import Path
from threading import Event, Lock
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		while chunk := stream.read(1024**2):
			digest.update(chunk)
	return digest.hexdigest()


def download(
	url: str,
	target: Path,
	*,
	sha256: str | None = None,
	timeout: float = 30,
	connections: int = 1,
	segment_bytes: int = 8 * 1024**2,
	retries: int = 3,
) -> dict:
	"""Preserve partial transfers; never publish a truncated or checksum-bad file.

	Resume requires a matching URL fingerprint and an HTTP validator. If the
	server ignores Range/If-Range, replace the partial transfer rather than append.
	No URL credentials or query parameters are stored in transfer metadata.
	"""
	target = Path(target)
	target.parent.mkdir(parents=True, exist_ok=True)
	if not 1 <= connections <= 16 or segment_bytes < 1 or timeout <= 0 or retries < 0:
		raise ValueError("Use 1–16 connections and positive segment/timeout limits")
	if sha256 is not None and (len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256)):
		raise ValueError("SHA-256 must contain 64 hexadecimal digits")
	if target.exists():
		digest = file_sha256(target)
		if sha256 is not None and digest != sha256.lower():
			raise ValueError("Existing archive checksum does not match")
		return {"bytes": target.stat().st_size, "sha256": digest, "resumed_bytes": 0}
	if connections > 1:
		try:
			with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
				size = int(response.headers.get("Content-Length", 0))
				validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
				if response.headers.get("Accept-Ranges") == "bytes" and validator and size > segment_bytes:
					return _segments(url, target, size, validator, sha256, timeout, connections, segment_bytes, retries)
		except HTTPError as error:
			if error.code not in (405, 501):
				raise
			error.close()
			# Some public servers support GET but deliberately omit HEAD.
			# Use the same atomic sequential path when range discovery is unavailable.
	partial = target.with_name(target.name + ".part")
	metadata = target.with_name(target.name + ".part.json")
	url_id = hashlib.sha256(url.encode()).hexdigest()
	previous = json.loads(metadata.read_text()) if metadata.exists() else {}
	validator = previous.get("validator") if previous.get("url_id") == url_id else None
	offset = partial.stat().st_size if partial.exists() and validator else 0
	headers = {"Range": f"bytes={offset}-", "If-Range": str(validator)} if offset else {}
	try:
		response = urlopen(Request(url, headers=headers), timeout=timeout)
	except HTTPError as error:
		if error.code != 416 or not offset:
			raise
		error.close()
		offset = 0
		response = urlopen(Request(url), timeout=timeout)
	with response:
		if offset and response.status == 206:
			if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
				raise ValueError("Server returned an inconsistent resume offset")
		else:
			offset = 0
		total = response.headers.get("Content-Length")
		total = offset + int(total) if total is not None else None
		validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
		metadata.write_text(json.dumps({"url_id": url_id, "validator": validator}))
		last_log = monotonic()
		with partial.open("ab" if offset else "wb") as stream:
			while chunk := response.read(1024**2):
				stream.write(chunk)
				if monotonic() - last_log >= 30:
					logging.info("Download: %.1f MiB received", stream.tell() / 1024**2)
					last_log = monotonic()
			stream.flush()
			os.fsync(stream.fileno())
		if total is not None and partial.stat().st_size != total:
			raise OSError("Archive transfer ended before its declared length")
	digest = file_sha256(partial)
	if sha256 is not None and digest != sha256.lower():
		partial.unlink()
		metadata.unlink(missing_ok=True)
		raise ValueError("Downloaded archive checksum does not match")
	os.replace(partial, target)
	metadata.unlink(missing_ok=True)
	return {"bytes": target.stat().st_size, "sha256": digest, "resumed_bytes": offset}


def _segments(url, target, size, validator, sha256, timeout, connections, segment_bytes, retries) -> dict:
	"""Independently resumable fixed ranges; assemble only verified-length segments."""
	folder = target.with_name(target.name + ".segments")
	folder.mkdir(exist_ok=True)
	identity = {
		"url_id": hashlib.sha256(url.encode()).hexdigest(),
		"bytes": size,
		"validator": validator,
		"segment_bytes": segment_bytes,
	}
	metadata = folder / "transfer.json"
	if metadata.exists() and json.loads(metadata.read_text()) != identity:
		raise ValueError("Segmented source changed; choose a new download directory")
	metadata.write_text(json.dumps(identity))
	partial = target.with_name(target.name + ".part")
	partial_metadata = target.with_name(target.name + ".part.json")
	first = folder / "000000"
	if partial.exists() and partial_metadata.exists() and not first.exists():
		old = json.loads(partial_metadata.read_text())
		if old.get("url_id") == identity["url_id"] and old.get("validator") == validator:
			with partial.open("rb") as source, first.open("wb") as output:
				remaining = min(partial.stat().st_size, segment_bytes)
				while remaining:
					chunk = source.read(min(remaining, 1024**2))
					if not chunk:
						raise OSError("Partial transfer changed while seeding segments")
					output.write(chunk)
					remaining -= len(chunk)
	ranges = [
		(index, start, min(size, start + segment_bytes)) for index, start in enumerate(range(0, size, segment_bytes))
	]
	resumed = sum(path.stat().st_size for path in folder.iterdir() if path.name.isdigit())
	progress = {"bytes": resumed, "last_log": monotonic(), "retries": 0}
	lock = Lock()
	cancelled = Event()

	def transfer_once(item):
		if cancelled.is_set():
			raise RuntimeError("Transfer cancelled; cached ranges can be resumed")
		index, start, stop = item
		path = folder / f"{index:06d}"
		offset = path.stat().st_size if path.exists() else 0
		if offset > stop - start:
			raise ValueError("Cached segment exceeds its range")
		if offset == stop - start:
			return path
		headers = {"Range": f"bytes={start + offset}-{stop - 1}", "If-Range": validator}
		with urlopen(Request(url, headers=headers), timeout=timeout) as response:
			expected_range = f"bytes {start + offset}-{stop - 1}/{size}"
			if response.status != 206 or response.headers.get("Content-Range") != expected_range:
				raise ValueError("Server did not preserve the requested byte range")
			with path.open("ab") as output:
				while chunk := response.read(1024**2):
					if cancelled.is_set():
						raise RuntimeError("Transfer cancelled; cached ranges can be resumed")
					if output.tell() + len(chunk) > stop - start:
						raise ValueError("Server exceeded the requested byte range")
					output.write(chunk)
					with lock:
						progress["bytes"] += len(chunk)
						if monotonic() - progress["last_log"] >= 30:
							logging.info("Download: %.1f / %.1f MiB", progress["bytes"] / 1024**2, size / 1024**2)
							progress["last_log"] = monotonic()
		if path.stat().st_size != stop - start:
			raise OSError("Incomplete download segment; rerun to resume")
		return path

	def transfer(item):
		for attempt in range(retries + 1):
			try:
				return transfer_once(item)
			except (OSError, HTTPException) as error:
				if (
					attempt == retries
					or cancelled.is_set()
					or getattr(error, "errno", None) in (13, 28)
					or isinstance(error, HTTPError)
					and error.code not in (408, 429, 500, 502, 503, 504)
				):
					raise
				with lock:
					progress["retries"] += 1
				logging.warning("Retrying interrupted range %d, attempt %d", item[0], attempt + 1)
				sleep(min(8, 2**attempt))
		raise AssertionError("Unreachable retry state")

	logging.info("Downloading with %d connections; %.1f MiB cached", connections, resumed / 1024**2)
	with ThreadPoolExecutor(max_workers=connections) as executor:
		pending = {executor.submit(transfer, item): item[0] for item in ranges}
		paths = {}
		try:
			for future in as_completed(pending):
				paths[pending[future]] = future.result()
		except BaseException:
			cancelled.set()
			for future in pending:
				future.cancel()
			raise
	assembled = target.with_name(target.name + ".assembled")
	digest = hashlib.sha256()
	with assembled.open("wb") as output:
		for index in range(len(ranges)):
			path = paths[index]
			with path.open("rb") as source:
				while chunk := source.read(1024**2):
					output.write(chunk)
					digest.update(chunk)
		output.flush()
		os.fsync(output.fileno())
	if sha256 is not None and digest.hexdigest() != sha256.lower():
		assembled.unlink()
		raise ValueError("Downloaded archive checksum does not match")
	os.replace(assembled, target)
	shutil.rmtree(folder)
	partial.unlink(missing_ok=True)
	partial_metadata.unlink(missing_ok=True)
	return {
		"bytes": size,
		"sha256": digest.hexdigest(),
		"resumed_bytes": resumed,
		"connections": connections,
		"range_retries": progress["retries"],
	}
