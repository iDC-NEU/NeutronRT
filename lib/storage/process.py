"""Local CPU storage actors; transport details stay below the topology facade."""

import mmap
import multiprocessing as mp
import os
import subprocess
import sys
from contextlib import ExitStack, suppress
from multiprocessing.reduction import recv_handle, send_handle
from threading import RLock
from time import perf_counter, process_time

import numpy as np
import psutil

from lib.storage.base import BaseTopology
from lib.storage.shard import TopologyShard


def _actor(connection, base, start, stop, overlay_bytes, response_bytes, profiling):
	operations: dict[str, int] = {}
	seconds: dict[str, float] = {}
	shared_bytes = 0
	try:
		shard = TopologyShard(base, start, stop, max_overlay_bytes=overlay_bytes)
		connection.send((True, None))
		while True:
			command, *args = connection.recv()
			if command == "close":
				break
			started = perf_counter() if profiling else 0.0
			try:
				if command in ("rows", "sample_rows"):
					direction, nodes = args[:2]
					fanout, seed = args[2:] if command == "sample_rows" else (-1, 0)
					counts = shard.rows[direction].degrees(nodes)
					if fanout != -1:
						if type(fanout) is not int or fanout < 1:
							raise ValueError("Fanout must be positive or -1")
						counts = np.minimum(counts, fanout)
					elements = nodes.size + int(counts.sum())
					if (elements * 8) > response_bytes:
						raise MemoryError("Selected adjacency exceeds the shared response budget")
					fd = os.memfd_create("neutronrt-selected-rows", os.MFD_CLOEXEC)
					try:
						os.ftruncate(fd, max(8, elements * 8))
						with mmap.mmap(fd, max(8, elements * 8)) as mapping:
							values = np.ndarray((elements,), dtype=np.int64, buffer=mapping)
							try:
								shard.fill(direction, nodes, values, fanout=fanout, seed=seed)
							finally:
								del values
						if profiling:
							shared_bytes += elements * 8
							operations[command] = operations.get(command, 0) + 1
							seconds[command] = seconds.get(command, 0.0) + perf_counter() - started
						connection.send((True, ("mapped-rows", nodes.size, elements - nodes.size)))
						send_handle(connection, fd, os.getppid())
					finally:
						os.close(fd)
					continue
				elif command == "degrees":
					result = shard.rows[args[0]].degrees(args[1])
				elif command == "exists":
					result = shard.rows["out"].exists(*args)
				elif command == "multiplicities":
					result = shard.rows["out"].multiplicities(*args)
				elif command == "prepare":
					result = shard.prepare(args[0], args[1], delete=args[2], revision=args[3])
				elif command == "commit":
					shard.commit(args[0])
					result = None
				elif command == "abort":
					shard.abort()
					result = None
				elif command == "stats":
					torch_module = sys.modules.get("torch")
					memory = psutil.Process().memory_full_info()
					result = {
						"revision": shard.revision,
						"operations": dict(operations),
						"seconds": dict(seconds),
						"shared_response_bytes": shared_bytes,
						"cpu_seconds": process_time(),
						"rss_bytes": memory.rss,
						"pss_bytes": getattr(memory, "pss", None),
						"uss_bytes": getattr(memory, "uss", None),
						"overlay_bytes": sum(rows.bytes_used for rows in shard.rows.values()),
						"overlay_metadata_bytes": sum(rows.overlay_metadata_bytes for rows in shard.rows.values()),
						"overlay_rows": sum(len(rows.overlays) for rows in shard.rows.values()),
						"response_buffers": 0,
						"torch_loaded": torch_module is not None,
						"cuda_initialized": bool(torch_module and torch_module.cuda.is_initialized()),
					}
				elif command == "rss":
					result = {"rss_bytes": int(psutil.Process().memory_info().rss)}
				else:
					raise ValueError("Unknown local storage operation")
				if profiling and command not in ("stats", "rss"):
					operations[command] = operations.get(command, 0) + 1
					seconds[command] = seconds.get(command, 0.0) + perf_counter() - started
				connection.send((True, result))
			except Exception as error:
				connection.send((False, (type(error).__name__, str(error))))
	except (EOFError, BrokenPipeError):
		pass
	finally:
		connection.close()


class _Process:
	"""Lifecycle adapter for a clean interpreter, without re-importing caller main."""

	def __init__(self, process):
		self.process = process
		self.pid = process.pid

	def is_alive(self):
		return self.process.poll() is None

	def join(self, timeout):
		with suppress(subprocess.TimeoutExpired):
			self.process.wait(timeout=timeout)

	def terminate(self):
		self.process.terminate()


class LocalStorage:
	"""One vertex range per process, independent of model worker devices.

	Only paths/ranges and budgets enter a child at startup. Requests pickle node
	IDs, never the graph, features, embeddings, or adjacency responses. Neighbor
	data crosses the local process boundary in shared memory.
	"""

	def __init__(
		self,
		base: BaseTopology,
		partitions: int,
		*,
		overlay_bytes: int,
		response_bytes: int,
		timeout=30,
		profiling: bool = False,
	):
		if type(partitions) is not int or partitions < 1 or overlay_bytes < 0 or response_bytes < 0 or timeout <= 0:
			raise ValueError("Invalid local storage partitions, budgets, or timeout")
		self.base, self.timeout = base, timeout
		self.boundaries = np.array([base.nodes * i // partitions for i in range(partitions + 1)], dtype=np.int64)
		self.processes: list = []
		self.connections: list = []
		self.locks: list = []
		self.closed = False
		if not hasattr(os, "memfd_create"):
			raise NotImplementedError("The local shared-FD storage backend requires Linux memfd")
		try:
			for start, stop in zip(self.boundaries[:-1], self.boundaries[1:], strict=True):
				parent, child = mp.Pipe()
				try:
					process = _Process(
						subprocess.Popen(
							[sys.executable, "-m", "lib.storage.actor", str(child.fileno())],
							pass_fds=(child.fileno(),),
							env={
								**os.environ,
								"CUDA_VISIBLE_DEVICES": "",
								"OMP_NUM_THREADS": "1",
								"OPENBLAS_NUM_THREADS": "1",
							},
						)
					)
				except BaseException:
					parent.close()
					raise
				finally:
					child.close()
				self.processes.append(process)
				self.connections.append(parent)
				self.locks.append(RLock())
				parent.send((base, int(start), int(stop), overlay_bytes, response_bytes, profiling))
			for index in range(partitions):
				self._receive(index)
		except BaseException:
			self.close()
			raise

	def _receive(self, index):
		connection = self.connections[index]
		if not connection.poll(self.timeout):
			raise TimeoutError("Local storage process did not respond")
		valid, result = connection.recv()
		if not valid:
			kind, message = result
			error = {"ValueError": ValueError, "MemoryError": MemoryError, "KeyError": KeyError}.get(kind, RuntimeError)
			raise error(message)
		if isinstance(result, tuple) and len(result) == 3 and result[0] == "mapped-rows":
			result = (recv_handle(connection), result[1], result[2])
		return result

	def groups(self, nodes):
		if nodes.dtype != np.int64 or nodes.ndim != 1:
			raise ValueError("Storage node IDs must be one-dimensional int64 arrays")
		if nodes.size and (nodes.min() < 0 or nodes.max() >= self.base.nodes):
			raise ValueError("Storage node ID out of range")
		owners = np.searchsorted(self.boundaries[1:], nodes, side="right")
		return {int(owner): np.flatnonzero(owners == owner) for owner in np.unique(owners)}

	def exchange(self, requests, *, results=None):
		"""Post to all selected partitions before receiving; drain every reply."""
		if self.closed:
			raise RuntimeError("Local storage is closed")
		results = {} if results is None else results
		errors = []
		for index, request in requests.items():
			self.connections[index].send(request)
		for index in requests:
			try:
				results[index] = self._receive(index)
			except Exception as error:
				errors.append(error)
		if errors:
			raise errors[0]
		return results

	def locked(self, indices):
		stack = ExitStack()
		for index in sorted(indices):
			stack.enter_context(self.locks[index])
		return stack

	def stats(self):
		with self.locked(range(len(self.processes))):
			return self.exchange(dict.fromkeys(range(len(self.processes)), ("stats",)))

	def sample_rss(self, *, timeout=None):
		"""Read only RSS for a profile sampler; avoid a full stats payload per tick."""
		with self.locked(range(len(self.processes))):
			return self.exchange(dict.fromkeys(range(len(self.processes)), ("rss",)))

	def close(self):
		if self.closed:
			return
		self.closed = True
		for connection, process in zip(self.connections, self.processes, strict=True):
			if process.is_alive():
				with suppress(OSError, EOFError):
					connection.send(("close",))
		for connection, process in zip(self.connections, self.processes, strict=True):
			process.join(min(self.timeout, 5))
			if process.is_alive():
				process.terminate()
				process.join(5)
			connection.close()
