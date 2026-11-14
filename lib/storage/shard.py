"""Own incoming and outgoing rows of a CPU vertex partition."""

import numpy as np

from lib.storage.base import BaseTopology
from lib.storage.rows import MappedRows


class TopologyShard:
	def __init__(self, base: BaseTopology, start: int, stop: int, *, max_overlay_bytes: int):
		self.rows = {
			direction: MappedRows(*base.files(direction), start, stop, max_overlay_bytes=max_overlay_bytes)
			for direction in ("in", "out")
		}
		self.revision = 0
		self.pending: dict = {}

	def prepare(self, outgoing: tuple, incoming: tuple, *, delete: bool, revision: int) -> tuple[int, int]:
		if revision != self.revision + 1 or self.pending:
			raise ValueError("Invalid storage mutation revision")
		try:
			for direction, pairs in (("out", outgoing), ("in", incoming)):
				self.pending[direction] = self.rows[direction].prepare(*pairs, delete=delete)
		except BaseException:
			self.abort()
			raise
		return self.pending["out"].edge_delta, self.pending["in"].edge_delta

	def commit(self, revision: int) -> None:
		if revision != self.revision + 1 or set(self.pending) != {"in", "out"}:
			raise ValueError("No matching prepared storage mutation")
		for direction in ("in", "out"):
			self.rows[direction].commit(self.pending[direction])
		self.pending.clear()
		self.revision = revision

	def abort(self) -> None:
		for rows in self.rows.values():
			rows.abort()
		self.pending.clear()

	def fill(self, direction: str, nodes: np.ndarray, buffer: np.ndarray, *, fanout: int = -1, seed: int = 0) -> None:
		rows = self.rows[direction]
		counts = rows.degrees(nodes)
		if fanout != -1:
			if type(fanout) is not int or fanout < 1:
				raise ValueError("Fanout must be positive or -1")
			rows.sample_into(nodes, counts, buffer, fanout=fanout, seed=seed)
			return
		if buffer.shape != (nodes.size + int(counts.sum()),):
			raise ValueError("Shared response does not match the selected adjacency rows")
		buffer[: nodes.size] = counts
		offset = nodes.size
		for node, size in zip(nodes, counts, strict=True):
			buffer[offset : offset + size] = rows.row(int(node))
			offset += int(size)
