"""CPU topology facade for affected analysis and full-neighbor block planning."""

import mmap
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable, Optional, TypeVar, cast

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from torch import Tensor

from lib.storage.base import BaseTopology
from lib.storage.process import LocalStorage

Result = TypeVar("Result")
Rows = list[tuple[np.ndarray, np.ndarray, np.ndarray]]


class ShardedTopology:
	device = torch.device("cpu")

	def __init__(
		self,
		graph: DGLGraph,
		partitions: int,
		*,
		directory: Optional[str] = None,
		overlay_bytes: int = 256 * 1024**2,
		response_bytes: int = 256 * 1024**2,
		timeout: float = 30,
		profiling: bool = False,
	) -> None:
		if directory is not None:
			Path(directory).mkdir(parents=True, exist_ok=True)
		self._folder = TemporaryDirectory(prefix="neutronrt-topology-", dir=directory)
		self.response_bytes = response_bytes
		started = perf_counter()
		try:
			self.base = BaseTopology.export(graph, Path(self._folder.name))
			export_done = perf_counter()
			self.storage = LocalStorage(
				self.base,
				partitions,
				overlay_bytes=overlay_bytes,
				response_bytes=response_bytes,
				timeout=timeout,
				profiling=profiling,
			)
		except BaseException:
			self._folder.cleanup()
			raise
		self.initialization = {
			"export_seconds": export_done - started,
			"actor_startup_seconds": perf_counter() - export_done,
			"base_file_bytes": sum(
				path.stat().st_size for direction in ("in", "out") for path in self.base.files(direction)
			),
		}
		self.revision, self.edges = 0, self.base.edges

	def num_nodes(self) -> int:
		return self.base.nodes

	def num_edges(self) -> int:
		return self.edges

	def create_formats_(self) -> None:
		pass

	@staticmethod
	def _nodes(nodes: Tensor) -> np.ndarray:
		if (
			not isinstance(nodes, torch.Tensor)
			or nodes.device.type != "cpu"
			or nodes.dtype != torch.int64
			or nodes.ndim != 1
		):
			raise ValueError("Topology requests require one-dimensional CPU int64 node IDs")
		return nodes.numpy()

	def _metadata(
		self, command: str, nodes: Tensor, *, direction: Optional[str] = None, others: Optional[np.ndarray] = None
	) -> Tensor:
		values = self._nodes(nodes)
		groups = self.storage.groups(values)
		result: np.ndarray = np.empty(values.size, dtype=np.bool_ if command == "exists" else np.int64)
		requests = {}
		if direction is None and others is None:
			raise ValueError("Edge existence requests require both endpoints")
		for index, positions in groups.items():
			requests[index] = (
				(command, direction, values[positions])
				if direction is not None
				else (command, values[positions], others[positions] if others is not None else None)
			)
		with self.storage.locked(groups):
			for index, response in self.storage.exchange(requests).items():
				result[groups[index]] = response
		return torch.from_numpy(result)

	def in_degrees(self, nodes: Optional[Tensor] = None) -> Tensor:
		return self._metadata("degrees", torch.arange(self.num_nodes()) if nodes is None else nodes, direction="in")

	def out_degrees(self, nodes: Optional[Tensor] = None) -> Tensor:
		return self._metadata("degrees", torch.arange(self.num_nodes()) if nodes is None else nodes, direction="out")

	def _edge_metadata(self, command: str, sources: Tensor, targets: Tensor) -> Tensor:
		other = self._nodes(targets)
		if other.shape != self._nodes(sources).shape:
			raise ValueError("Topology edge endpoints must have matching shapes")
		self.storage.groups(other)
		return self._metadata(command, sources, others=other)

	def has_edges_between(self, sources: Tensor, targets: Tensor) -> Tensor:
		return self._edge_metadata("exists", sources, targets)

	def edge_multiplicities(self, sources: Tensor, targets: Tensor) -> Tensor:
		"""Count current edges; a mutation caller holds the graph fence until deletion."""
		return self._edge_metadata("multiplicities", sources, targets)

	def _selected(
		self, direction: str, nodes: Tensor, consume: Callable[[Rows], Result], *, fanout: int = -1, seed: int = 0
	) -> Result:
		"""Borrow local response buffers only for the duration of the consumer.

		The consumer must return owned results, such as compiled block indices;
		shared views must never escape this scope.
		"""
		values = self._nodes(nodes)
		groups = self.storage.groups(values)
		descriptors: dict[int, tuple[int, int, int]] = {}
		with self.storage.locked(groups), ExitStack() as stack:
			try:
				degrees = self.storage.exchange({i: ("degrees", direction, values[p]) for i, p in groups.items()})
				if fanout != -1:
					degrees = {index: np.minimum(counts, fanout) for index, counts in degrees.items()}
				elements = values.size + sum(int(count) for counts in degrees.values() for count in counts)
				if elements * 8 > self.response_bytes:
					raise MemoryError("Selected adjacency exceeds the shared response budget")
				requests = {
					i: ("rows", direction, values[p])
					if fanout == -1
					else ("sample_rows", direction, values[p], fanout, seed)
					for i, p in groups.items()
				}
				self.storage.exchange(requests, results=descriptors)
				arrays = []
				for index, (fd, count, edges) in descriptors.items():
					mapping = mmap.mmap(fd, max(8, (count + edges) * 8), access=mmap.ACCESS_READ)
					stack.callback(mapping.close)
					array = np.ndarray((count + edges,), dtype=np.int64, buffer=mapping)
					array.flags.writeable = False
					arrays.append((groups[index], array[:count], array[count:]))
				return consume(arrays)
			finally:
				# The kernel frees each anonymous buffer after its last FD/mapping closes.
				for fd, _, _ in descriptors.values():
					os.close(fd)

	def expand(self, frontier: Tensor, edge_nodes: Tensor) -> Tensor:
		self.storage.groups(self._nodes(edge_nodes))
		degrees = self.out_degrees(frontier)
		if (frontier.numel() + int(degrees.sum())) * 8 > self.response_bytes:
			if frontier.numel() <= 1:
				raise MemoryError("A single outgoing row exceeds the shared response budget")
			middle = frontier.numel() // 2
			left = self.expand(frontier[:middle], edge_nodes)
			right = self.expand(frontier[middle:], torch.empty(0, dtype=torch.int64))
			return cast(Tensor, torch.cat((left, right)).unique(sorted=True))
		return self._selected(
			"out",
			frontier,
			lambda groups: torch.from_numpy(
				np.unique(
					np.concatenate(
						[self._nodes(frontier), self._nodes(edge_nodes), *(neighbors for _, _, neighbors in groups)]
					)
				)
			),
		)

	def compile_block(self, targets: Tensor) -> DGLBlock:
		return self._compile_block(targets)

	def sample_block(self, targets: Tensor, *, fanout: int, seed: int) -> DGLBlock:
		"""Sample edge positions inside storage; never transfer full high-degree rows."""
		if type(fanout) is not int or fanout < 1 or type(seed) is not int or seed < 0:
			raise ValueError("Sampling requires a positive fanout and nonnegative integer seed")
		return self._compile_block(targets, fanout=fanout, seed=seed)

	def _compile_block(self, targets: Tensor, *, fanout: int = -1, seed: int = 0) -> DGLBlock:
		self.storage.groups(self._nodes(targets))
		if targets.unique().numel() != targets.numel():
			raise ValueError("Block targets must be unique")

		def build(groups: Rows) -> DGLBlock:
			# Concatenation creates only the selected block's owned edge metadata.
			neighbors = np.concatenate([values for _, _, values in groups]) if groups else np.empty(0, dtype=np.int64)
			ends = (
				np.concatenate([np.repeat(positions, counts) for positions, counts, _ in groups])
				if groups
				else np.empty(0, dtype=np.int64)
			)
			extra = np.setdiff1d(np.unique(neighbors), self._nodes(targets), assume_unique=True)
			sources = np.concatenate((self._nodes(targets), extra))
			order = np.argsort(sources)
			starts = order[np.searchsorted(sources[order], neighbors)]
			block = dgl.create_block(
				(torch.from_numpy(starts), torch.from_numpy(ends)),
				num_src_nodes=sources.size,
				num_dst_nodes=targets.numel(),
			)
			block.srcdata[dgl.NID] = torch.from_numpy(sources)
			block.dstdata[dgl.NID] = targets
			return block

		return self._selected("in", targets, build, fanout=fanout, seed=seed)

	def out_edges(self, sources: Tensor) -> tuple[Tensor, Tensor]:
		"""Return owned selected edge IDs; repeated graph edges remain distinct.

		This interface accepts unique source nodes. It transfers only the selected
		adjacency metadata, never an embedding vector per edge.
		"""
		self.storage.groups(self._nodes(sources))
		if sources.unique().numel() != sources.numel():
			raise ValueError("Outgoing row sources must be unique")

		def build(groups: Rows) -> tuple[Tensor, Tensor]:
			if not groups:
				return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
			starts = np.concatenate(
				[np.repeat(self._nodes(sources)[positions], counts) for positions, counts, _ in groups]
			)
			ends = np.concatenate([neighbors for _, _, neighbors in groups])
			return torch.from_numpy(starts), torch.from_numpy(ends)

		return self._selected("out", sources, build)

	def _mutate(self, sources: Tensor, targets: Tensor, *, delete: bool) -> None:
		first, second = self._nodes(sources), self._nodes(targets)
		if first.shape != second.shape:
			raise ValueError("Topology edge endpoints must have matching shapes")
		outgoing, incoming = self.storage.groups(first), self.storage.groups(second)
		indices = range(len(self.storage.processes))
		requests = {}
		empty = np.empty(0, dtype=np.int64)
		for index in indices:
			u, v = outgoing.get(index), incoming.get(index)
			requests[index] = (
				"prepare",
				(first[u], second[u]) if u is not None else (empty, empty),
				(second[v], first[v]) if v is not None else (empty, empty),
				delete,
				self.revision + 1,
			)
		with self.storage.locked(indices):
			try:
				deltas = self.storage.exchange(requests)
				out_delta = sum(value[0] for value in deltas.values())
				if out_delta != sum(value[1] for value in deltas.values()):
					raise RuntimeError("Incoming and outgoing topology indexes disagree")
			except BaseException:
				self.storage.exchange(dict.fromkeys(indices, ("abort",)))
				raise
			self.storage.exchange(dict.fromkeys(indices, ("commit", self.revision + 1)))
			self.revision += 1
			self.edges += out_delta

	def add_edges(self, sources: Tensor, targets: Tensor) -> None:
		self._mutate(sources, targets, delete=False)

	def delete_pairs(self, sources: Tensor, targets: Tensor) -> None:
		self._mutate(sources, targets, delete=True)

	def close(self) -> None:
		self.storage.close()
		self._folder.cleanup()

	def __enter__(self):
		return self

	def __exit__(self, *_):
		self.close()
