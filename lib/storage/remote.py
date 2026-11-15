"""Authenticated network storage client for one topology partition per server."""

from __future__ import annotations

from contextlib import ExitStack, suppress
from multiprocessing.connection import Client
from threading import RLock

import numpy as np

from lib.storage.capacity import StorageNodeBudget


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

	def _receive(self):
		if not self.connection.poll(self.timeout):
			raise TimeoutError(f"Storage server {self.endpoint} did not respond")
		ok, value = self.connection.recv()
		if not ok:
			kind, message = value
			error = {"ValueError": ValueError, "MemoryError": MemoryError, "KeyError": KeyError}.get(kind, RuntimeError)
			raise error(message)
		return value

	def request(self, command, *args):
		with self.lock:
			self.connection.send((command, *args))
			return self._receive()

	def close(self) -> None:
		with self.lock:
			with suppress(BrokenPipeError, EOFError, OSError):
				self.connection.send(("close",))
			self.connection.close()


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
		self.boundaries = np.array(
			[base.nodes * index // len(endpoints) for index in range(len(endpoints) + 1)], dtype=np.int64
		)
		self.processes = [None] * len(endpoints)
		self.connections = []
		self.locks = []
		self.closed = False
		try:
			for endpoint, (start, stop), budget in zip(
				endpoints,
				zip(self.boundaries[:-1], self.boundaries[1:], strict=True),
				budgets,
				strict=True,
			):
				connection = _RemoteConnection(endpoint, authkey, timeout)
				connection.request(
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
				)
				self.connections.append(connection)
				self.locks.append(connection.lock)
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

	def close(self):
		if self.closed:
			return
		self.closed = True
		for connection in self.connections:
			connection.close()
