"""Optional zero-copy native delta kernel; all large inputs remain tensor-owned."""

import ctypes
import subprocess
from pathlib import Path
from threading import Lock

import torch


def compile_native(directory):
	source = Path(__file__).with_name("csrc") / "parallel.cpp"
	library = Path(directory) / "parallel_graph.so"
	subprocess.run(
		["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-pthread", str(source), "-o", str(library)], check=True
	)
	return library


class NativeDeltaAccumulator:
	def __init__(self, graph, library):
		if graph.device.type != "cpu" or graph.idtype != torch.int64 or not graph.is_homogeneous:
			raise ValueError("Native delta accumulation requires an int64 homogeneous CPU graph")
		self.offsets, self.sources, _ = graph.adj_tensors("csc")
		self.graph = graph
		self._graph_index = graph._graph
		self._lock = Lock()
		self.node_count = graph.num_nodes()
		self.row_map = torch.full((self.node_count,), -1, dtype=torch.int64)
		self.previous_changed = torch.empty(0, dtype=torch.int64)
		self.library = ctypes.CDLL(str(library))
		pointer, integer = ctypes.c_void_p, ctypes.c_int64
		self.kernel = self.library.pull_feature_delta
		self.kernel.argtypes = [
			pointer,
			pointer,
			pointer,
			integer,
			pointer,
			pointer,
			integer,
			integer,
			pointer,
			integer,
			integer,
			pointer,
			integer,
			integer,
		]
		self.kernel.restype = ctypes.c_int

	def apply(self, current, previous, changed, targets, aggregates, *, threads=1):
		# CDLL releases the GIL. Serialize the reusable row map while native
		# workers partition independent output rows within each call.
		with self._lock:
			self._apply(current, previous, changed, targets, aggregates, threads=threads)

	def _apply(self, current, previous, changed, targets, aggregates, *, threads):
		if self.graph._graph is not self._graph_index:
			raise ValueError("Graph topology changed; rebuild the native adjacency view")
		if threads < 1:
			raise ValueError("Native thread count must be positive")
		for tensor in (current, previous, changed, targets, aggregates):
			if tensor.device.type != "cpu":
				raise ValueError("Native delta inputs must be on CPU")
		if current.ndim != 2 or current.shape[0] != self.node_count:
			raise ValueError("Current embeddings must match graph nodes")
		if current.dtype != torch.float32 or previous.dtype != torch.float32:
			raise ValueError("Native source embeddings must be float32")
		if aggregates.shape != current.shape or aggregates.dtype != torch.float64 or not aggregates.is_contiguous():
			raise ValueError("Native aggregate output must be a contiguous float64 matrix")
		if previous.shape != (changed.numel(), current.shape[1]):
			raise ValueError("Previous rows do not match the changed sources")
		for nodes in (changed, targets):
			if nodes.ndim != 1 or nodes.dtype != torch.int64 or not nodes.is_contiguous():
				raise ValueError("Native node IDs must be contiguous int64 vectors")
			if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.node_count):
				raise ValueError("Native node ID out of range")
			if nodes.unique().numel() != nodes.numel():
				raise ValueError("Native node IDs must be unique")
		self.row_map[self.previous_changed] = -1
		self.row_map[changed] = torch.arange(changed.numel())
		self.previous_changed = changed.clone()
		status = self.kernel(
			self.offsets.data_ptr(),
			self.sources.data_ptr(),
			targets.data_ptr(),
			targets.numel(),
			self.row_map.data_ptr(),
			current.data_ptr(),
			current.stride(0),
			current.stride(1),
			previous.data_ptr(),
			previous.stride(0),
			previous.stride(1),
			aggregates.data_ptr(),
			current.shape[1],
			threads,
		)
		if status != 0:
			raise RuntimeError(f"Native delta kernel failed with status {status}")
