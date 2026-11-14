"""Borrow immutable mapped adjacency rows and stage sparse, bounded mutations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np


@dataclass(frozen=True)
class PreparedRows:
	rows: Dict[int, np.ndarray]
	bytes_used: int
	edge_delta: int
	visible: Dict[int, np.ndarray]
	index: np.ndarray
	degrees: np.ndarray


class MappedRows:
	"""Own one vertex interval; no private copy of the base adjacency arrays.

	Returned rows are read-only. A prepared mutation owns replacements only for
	changed rows, and becomes visible only on commit. Process/network recovery is
	outside the first local distribution emulation.
	"""

	def __init__(
		self, offsets: Path, neighbors: Path, start: int, stop: int, *, max_overlay_bytes: int = 256 * 1024**2
	) -> None:
		self.offsets = np.load(offsets, mmap_mode="r", allow_pickle=False)
		self.neighbors = np.load(neighbors, mmap_mode="r", allow_pickle=False)
		if (
			self.offsets.dtype != np.int64
			or self.neighbors.dtype != np.int64
			or self.offsets.ndim != 1
			or self.neighbors.ndim != 1
			or not 0 <= start <= stop < self.offsets.size
			or max_overlay_bytes < 0
		):
			raise ValueError("Invalid mapped adjacency layout, vertex interval, or overlay budget")
		if (
			self.offsets[0] != 0
			or self.offsets[-1] != self.neighbors.size
			or np.any(self.offsets[start + 1 : stop + 1] < self.offsets[start:stop])
		):
			raise ValueError("Invalid mapped adjacency offsets")
		self.start, self.stop = start, stop
		self.max_overlay_bytes = max_overlay_bytes
		self.overlays: Dict[int, np.ndarray] = {}
		self._overlay_index = np.empty(0, dtype=np.int64)
		self._overlay_degrees = np.empty(0, dtype=np.int64)
		self.bytes_used = 0
		self._pending: Optional[PreparedRows] = None

	def _ids(self, nodes: np.ndarray) -> None:
		if nodes.dtype != np.int64 or nodes.ndim != 1:
			raise ValueError("Storage row IDs must be one-dimensional int64 arrays")
		if nodes.size and (nodes.min() < self.start or nodes.max() >= self.stop):
			raise ValueError("Storage row is outside this partition")

	def row(self, node: int) -> np.ndarray:
		if not self.start <= node < self.stop:
			raise ValueError("Storage row is outside this partition")
		if node in self.overlays:
			return self.overlays[node]
		return cast(np.ndarray, self.neighbors[int(self.offsets[node]) : int(self.offsets[node + 1])])

	def degrees(self, nodes: np.ndarray) -> np.ndarray:
		self._ids(nodes)
		counts = self.offsets[nodes + 1] - self.offsets[nodes]
		positions, indices = self._overlay_positions(nodes)
		counts[positions] = self._overlay_degrees[indices]
		return cast(np.ndarray, counts)

	def _overlay_positions(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Route a query into immutable, commit-certified overlay metadata."""
		if not self._overlay_index.size:
			empty = np.empty(0, dtype=np.int64)
			return empty, empty
		indices = np.searchsorted(self._overlay_index, nodes)
		positions = np.flatnonzero(indices < self._overlay_index.size)
		positions = positions[self._overlay_index[indices[positions]] == nodes[positions]]
		return positions, indices[positions]

	@property
	def overlay_metadata_bytes(self) -> int:
		return self._overlay_index.nbytes + self._overlay_degrees.nbytes

	@staticmethod
	def _sample_positions(degrees: np.ndarray, fanout: int, generator: np.random.Generator) -> np.ndarray:
		chosen = np.empty((degrees.size, fanout), dtype=np.int64)
		for column in range(fanout):
			upper = degrees - fanout + column + 1
			candidate = generator.integers(upper)
			duplicate = (chosen[:, :column] == candidate[:, None]).any(axis=1)
			chosen[:, column] = np.where(duplicate, upper - 1, candidate)
		return chosen

	def sample_into(self, nodes, degrees, buffer, *, fanout: int, seed: int) -> None:
		"""Fill a bounded response; vectorize the small fanouts used by NS5/NS10.

		Floyd sampling chooses edge positions without replacement. Parallel graph
		edges therefore retain their multiplicity. Changed overlay rows are sampled
		from their committed replacement arrays, never from the stale mapped base.
		"""
		self._ids(nodes)
		counts = np.minimum(degrees, fanout)
		if buffer.shape != (nodes.size + int(counts.sum()),):
			raise ValueError("Sample response does not match selected row sizes")
		buffer[: nodes.size] = counts
		output = buffer[nodes.size :]
		starts = np.concatenate((np.zeros(1, dtype=np.int64), counts.cumsum()[:-1]))[: nodes.size]
		generator = np.random.default_rng(seed)
		if fanout > 32:
			for node, start, count in zip(nodes, starts, counts, strict=True):
				row = self.row(int(node))
				output[start : start + count] = generator.choice(row, int(count), replace=False)
			return
		overlay_positions, _ = self._overlay_positions(nodes)
		overlay = np.zeros(nodes.size, dtype=np.bool_)
		overlay[overlay_positions] = True
		for count in np.unique(degrees[(~overlay) & (degrees <= fanout)]):
			if not count:
				continue
			positions = np.flatnonzero((~overlay) & (degrees == count))
			columns = np.arange(count)
			output[starts[positions, None] + columns] = self.neighbors[self.offsets[nodes[positions], None] + columns]
		positions = np.flatnonzero((~overlay) & (degrees > fanout))
		if positions.size:
			chosen = self._sample_positions(degrees[positions], fanout, generator)
			output[starts[positions, None] + np.arange(fanout)] = self.neighbors[
				self.offsets[nodes[positions], None] + chosen
			]
		positions = np.flatnonzero(overlay & (degrees > fanout))
		chosen = self._sample_positions(degrees[positions], fanout, generator)
		for position, columns in zip(positions, chosen, strict=True):
			start = starts[position]
			output[start : start + fanout] = self.overlays[int(nodes[position])][columns]
		for position in np.flatnonzero(overlay & (degrees <= fanout)):
			row = self.overlays[int(nodes[position])]
			start, count = starts[position], int(counts[position])
			output[start : start + count] = row

	def exists(self, nodes: np.ndarray, others: np.ndarray) -> np.ndarray:
		self._pairs(nodes, others)
		return np.fromiter(
			(np.any(self.row(int(node)) == other) for node, other in zip(nodes, others, strict=True)),
			dtype=np.bool_,
			count=nodes.size,
		)

	def multiplicities(self, nodes: np.ndarray, others: np.ndarray) -> np.ndarray:
		"""Count parallel edges with bounded comparison scratch, including repeated requests."""
		self._pairs(nodes, others)
		return np.fromiter(
			(
				sum(int(np.count_nonzero(part == other)) for part in self._chunks(self.row(int(node))))
				for node, other in zip(nodes, others, strict=True)
			),
			dtype=np.int64,
			count=nodes.size,
		)

	def _pairs(self, nodes: np.ndarray, others: np.ndarray) -> None:
		self._ids(nodes)
		if others.dtype != np.int64 or others.shape != nodes.shape:
			raise ValueError("Storage edge endpoints must be matching int64 arrays")
		if others.size and (others.min() < 0 or others.max() >= self.offsets.size - 1):
			raise ValueError("Neighbor ID is outside the graph")

	def prepare(self, nodes: np.ndarray, others: np.ndarray, *, delete: bool) -> PreparedRows:
		"""Allocate changed rows before committing any part of an update."""
		if self._pending is not None:
			raise RuntimeError("A storage mutation is already prepared")
		self._pairs(nodes, others)
		changes: Dict[int, list] = {}
		for node, other in zip(nodes, others, strict=True):
			changes.setdefault(int(node), []).append(int(other))
		replacements = {}
		bytes_used, edge_delta = self.bytes_used, 0
		for node, values in changes.items():
			old = self.row(node)
			selected = np.asarray(values, dtype=np.int64)
			previous = self.overlays.get(node)
			previous_bytes = previous.nbytes if previous is not None else 0
			if delete:
				# Two bounded passes avoid a degree-sized temporary mask and reject
				# oversized replacements before allocating them.
				kept = sum(int(np.count_nonzero(~np.isin(part, selected))) for part in self._chunks(old))
				if kept == old.size:
					continue
				new_bytes = kept * old.itemsize
			else:
				new_bytes = old.nbytes + selected.nbytes
			if bytes_used + new_bytes - previous_bytes > self.max_overlay_bytes:
				raise MemoryError("Changed adjacency rows exceed this storage partition's overlay budget")
			if delete:
				new = np.empty(kept, dtype=np.int64)
				offset = 0
				for part in self._chunks(old):
					remaining = part[~np.isin(part, selected)]
					new[offset : offset + remaining.size] = remaining
					offset += remaining.size
			else:
				new = np.concatenate((old, selected))
			new.flags.writeable = False
			bytes_used += new.nbytes - previous_bytes
			edge_delta += new.size - old.size
			replacements[node] = new
		if bytes_used > self.max_overlay_bytes:
			raise MemoryError("Changed adjacency rows exceed this storage partition's overlay budget")
		visible = self.overlays.copy()
		visible.update(replacements)
		index = np.array(sorted(visible), dtype=np.int64)
		degrees = np.fromiter((visible[int(node)].size for node in index), dtype=np.int64, count=index.size)
		index.flags.writeable = degrees.flags.writeable = False
		prepared = PreparedRows(replacements, bytes_used, edge_delta, visible, index, degrees)
		self._pending = prepared
		return prepared

	@staticmethod
	def _chunks(values: np.ndarray):
		for start in range(0, values.size, 65536):
			yield values[start : start + 65536]

	def commit(self, prepared: PreparedRows) -> None:
		if prepared is not self._pending:
			raise ValueError("Storage commit does not match its prepared mutation")
		self.overlays = prepared.visible
		self._overlay_index = prepared.index
		self._overlay_degrees = prepared.degrees
		self.bytes_used = prepared.bytes_used
		self._pending = None

	def abort(self) -> None:
		self._pending = None
