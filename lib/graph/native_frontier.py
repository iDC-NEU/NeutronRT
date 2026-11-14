"""Optional GIL-free frontier expansion borrowing authoritative storage CSR."""

import ctypes
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Optional, Union

import torch
from dgl import DGLGraph
from torch import Tensor

from lib.graph.native import compile_native


class NativeFrontier:
	"""One byte of scratch per node; no graph or feature buffer copies.

	CSR views are refreshed after DGL replaces topology. The storage graph fence
	must exclude mutations during a call. Native threads operate only on borrowed
	Tensor allocations and return before those references may be released.
	"""

	def __init__(self, graph: DGLGraph, *, threads: int = 4, library: Optional[Union[str, Path]] = None) -> None:
		if graph.device.type != "cpu" or graph.idtype != torch.int64 or not graph.is_homogeneous:
			raise ValueError("Native analysis requires an int64 homogeneous CPU graph")
		if type(threads) is not int or threads < 1:
			raise ValueError("Native thread count must be a positive integer")
		self.graph, self.threads = graph, threads
		self._lock = Lock()
		self._folder = TemporaryDirectory(prefix="neutronrt-native-") if library is None else None
		if library is None:
			assert self._folder is not None
			library = compile_native(self._folder.name)
		self.library = ctypes.CDLL(str(library))
		self.kernel = self.library.frontier_mark_parallel
		self.kernel.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_int64,
			ctypes.c_void_p,
			ctypes.c_int64,
		]
		self.kernel.restype = ctypes.c_int
		self.offsets, self.neighbors, _ = graph.adj_tensors("csr")
		self._graph_index = graph._graph

	def __call__(self, frontier: Tensor, extra: Tensor) -> Tensor:
		with self._lock:
			for value in (frontier, extra):
				if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1:
					raise ValueError("Native frontier IDs must be CPU int64 vectors")
				if value.numel() and (value.min() < 0 or value.max() >= self.graph.num_nodes()):
					raise ValueError("Native frontier node ID out of range")
			if self.graph._graph is not self._graph_index:
				self.offsets, self.neighbors, _ = self.graph.adj_tensors("csr")
				self._graph_index = self.graph._graph
			mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
			mask[frontier] = True
			mask[extra] = True
			contiguous = frontier.contiguous()
			status = self.kernel(
				self.offsets.data_ptr(),
				self.neighbors.data_ptr(),
				contiguous.data_ptr(),
				contiguous.numel(),
				mask.data_ptr(),
				self.threads,
			)
			if status:
				raise RuntimeError(f"Native frontier failed with status {status}")
			return mask.nonzero().flatten()
