"""Authenticated network storage client for one topology partition per server."""

from __future__ import annotations

from contextlib import ExitStack, suppress
from hashlib import sha256
from multiprocessing.connection import Client
from threading import RLock
from time import monotonic

import numpy as np

from lib.storage.capacity import StorageNodeBudget

_FINGERPRINT_CHUNK_BYTES = 8 * 1024**2


def _fingerprint_array(digest, name: str, values: np.ndarray) -> None:
	array = np.asarray(values)
	if not array.flags.c_contiguous:
		raise ValueError("Mapped topology requires contiguous int64 adjacency buffers")
	digest.update(name.encode("ascii"))
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
	byte_view = memoryview(array).cast("B")
	for start in range(0, len(byte_view), _FINGERPRINT_CHUNK_BYTES):
		digest.update(byte_view[start : start + _FINGERPRINT_CHUNK_BYTES])


def _valid_offsets(offsets: np.ndarray, neighbors: np.ndarray) -> bool:
	if (
		offsets.ndim != 1
		or neighbors.ndim != 1
		or offsets.dtype != np.int64
		or neighbors.dtype != np.int64
		or offsets.size == 0
		or offsets[0] != 0
		or offsets[-1] != neighbors.size
	):
		return False
	chunk_elements = max(1, _FINGERPRINT_CHUNK_BYTES // offsets.itemsize)
	for start in range(0, offsets.size - 1, chunk_elements):
		stop = min(offsets.size - 1, start + chunk_elements)
		if np.any(offsets[start + 1 : stop + 1] < offsets[start:stop]):
			return False
	return True


def base_fingerprint(base) -> str:
	"""Return a path-independent digest of both exported adjacency indexes."""
	digest = sha256()
	digest.update(np.asarray((base.nodes, base.edges), dtype=np.int64).tobytes())
	for direction in ("in", "out"):
		offsets_path, neighbors_path = base.files(direction)
		offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
		neighbors = np.load(neighbors_path, mmap_mode="r", allow_pickle=False)
		if (
			offsets.size != base.nodes + 1
			or neighbors.size != base.edges
			or not _valid_offsets(offsets, neighbors)
			or (neighbors.size and (neighbors.min() < 0 or neighbors.max() >= base.nodes))
		):
			raise ValueError(f"Invalid exported {direction} adjacency index")
		_fingerprint_array(digest, f"{direction}:offsets", offsets)
		_fingerprint_array(digest, f"{direction}:neighbors", neighbors)
	return digest.hexdigest()


def graph_fingerprint(graph) -> str:
	"""Return the same canonical digest for a homogeneous CPU int64 graph."""
	if graph.device.type != "cpu" or str(graph.idtype) != "torch.int64" or not graph.is_homogeneous:
		raise ValueError("Mapped topology requires a homogeneous CPU int64 graph")
	digest = sha256()
	nodes, edges = graph.num_nodes(), graph.num_edges()
	digest.update(np.asarray((nodes, edges), dtype=np.int64).tobytes())
	for direction, layout in (("in", "csc"), ("out", "csr")):
		view = graph.formats(layout)
		arrays = view.adj_tensors(layout)[:2]
		for name, values in zip(("offsets", "neighbors"), arrays, strict=True):
			array = values.numpy()
			if array.dtype != np.int64 or not array.flags.c_contiguous:
				raise ValueError("Mapped topology requires contiguous int64 adjacency buffers")
			_fingerprint_array(digest, f"{direction}:{name}", array)
	return digest.hexdigest()


class _RemoteConnection:
	def __init__(self, endpoint: str, authkey: bytes, timeout: float):
		host, separator, port = endpoint.rpartition(":")
		if not separator or not host or not port.isdigit():
			raise ValueError(f"Storage endpoint must be HOST:PORT: {endpoint!r}")
		self.endpoint, self.timeout = endpoint, timeout
		self.connection = Client((host, int(port)), authkey=authkey)
		self.lock = RLock()
		self.connection.send(("ping",))
		if self._receive() != "ready":
			raise ConnectionError(f"Storage server {endpoint} did not accept the connection")

	def _receive(self, timeout=None):
		if not self.connection.poll(self.timeout if timeout is None else timeout):
			raise TimeoutError(f"Storage server {self.endpoint} did not respond")
		ok, value = self.connection.recv()
		if not ok:
			kind, message = value
			error = {"ValueError": ValueError, "MemoryError": MemoryError, "KeyError": KeyError}.get(kind, RuntimeError)
			raise error(message)
		return value

	def request(self, command, *args, timeout=None):
		with self.lock:
			self.connection.send((command, *args))
			return self._receive(timeout)

	def close(self, timeout: float = 0.25) -> None:
		acquired = self.lock.acquire(timeout=max(0.0, timeout))
		try:
			if acquired:
				with suppress(BrokenPipeError, EOFError, OSError):
					self.connection.send(("close",))
			with suppress(OSError):
				self.connection.close()
		finally:
			if acquired:
				self.lock.release()


class RemoteStorage:
	"""Storage facade matching ``LocalStorage.exchange`` and ``groups``."""

	remote = True

	def __init__(
		self,
		base,
		endpoints,
		*,
		budgets: tuple[StorageNodeBudget, ...],
		authkey: bytes,
		overlay_bytes: int,
		response_bytes: int,
		timeout: float,
	):
		if len(endpoints) < 1:
			raise ValueError("At least one remote storage endpoint is required")
		if len(budgets) != len(endpoints):
			raise ValueError("One storage budget is required per remote endpoint")
		self.base, self.timeout = base, timeout
		self._sampling_timeout = min(timeout, 0.5)
		self.boundaries = np.array(
			[base.nodes * index // len(endpoints) for index in range(len(endpoints) + 1)], dtype=np.int64
		)
		self.processes = [None] * len(endpoints)
		self.connections = []
		self._sampling_connections = []
		self.locks = []
		self.closed = False
		fingerprint = base_fingerprint(base)
		try:
			for endpoint, (start, stop), budget in zip(
				endpoints,
				zip(self.boundaries[:-1], self.boundaries[1:], strict=True),
				budgets,
				strict=True,
			):
				connection = _RemoteConnection(endpoint, authkey, timeout)
				self.connections.append(connection)
				self.locks.append(connection.lock)
				init_args = (
					"init",
					str(base.folder),
					base.nodes,
					base.edges,
					int(start),
					int(stop),
					overlay_bytes,
					response_bytes,
					budget.capacity_bytes,
					budget.reserve_bytes,
					fingerprint,
				)
				connection.request(*init_args)
				# Sampling uses a dedicated connection so a bounded RSS timeout cannot
				# desynchronise a control/data connection.
				sampling = _RemoteConnection(endpoint, authkey, timeout)
				sampling.request(*init_args)
				self._sampling_connections.append(sampling)
		except BaseException:
			self.close()
			raise

	def groups(self, nodes):
		if nodes.dtype != np.int64 or nodes.ndim != 1:
			raise ValueError("Storage node IDs must be one-dimensional int64 arrays")
		if nodes.size and (nodes.min() < 0 or nodes.max() >= self.base.nodes):
			raise ValueError("Storage node ID out of range")
		owners = np.searchsorted(self.boundaries[1:], nodes, side="right")
		return {int(owner): np.flatnonzero(owners == owner) for owner in np.unique(owners)}

	def exchange(self, requests, *, results=None):
		if self.closed:
			raise RuntimeError("Remote storage is closed")
		results = {} if results is None else results
		for index, request in requests.items():
			results[index] = self.connections[index].request(*request)
		return results

	def locked(self, indices):
		stack = ExitStack()
		for index in sorted(indices):
			stack.enter_context(self.locks[index])
		return stack

	def stats(self):
		return self.exchange(dict.fromkeys(range(len(self.connections)), ("stats",)))

	def sample_rss(self, *, timeout=None):
		"""Read endpoint RSS over dedicated connections under one total deadline."""
		if self.closed:
			raise RuntimeError("Remote storage is closed")
		budget = self._sampling_timeout if timeout is None else min(float(timeout), self._sampling_timeout)
		deadline = monotonic() + max(0.01, budget)
		results = {}
		for index, connection in enumerate(self._sampling_connections):
			if connection is None:
				results[index] = {"error": "sampling connection unavailable"}
				continue
			remaining = deadline - monotonic()
			if remaining <= 0:
				results[index] = {"error": "RSS sampling deadline elapsed"}
				continue
			try:
				results[index] = connection.request("rss", timeout=remaining)
			except Exception as error:
				results[index] = {"error": f"{type(error).__name__}: {error}"}
				with suppress(OSError, EOFError):
					connection.close()
				self._sampling_connections[index] = None
		return results

	def close(self):
		if self.closed:
			return
		self.closed = True
		for connection in self.connections:
			connection.close()
		for connection in self._sampling_connections:
			if connection is not None:
				connection.close()
