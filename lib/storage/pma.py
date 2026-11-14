"""CPU packed-memory adjacency with density-controlled segment rebalancing.

Keys encode (row, neighbor); holes separate sorted elements. Inserts rebalance
the smallest ancestor with spare density, growing the root when necessary.
This reference implementation favors correctness over native traversal speed.
"""

import numpy as np
import torch


class PackedAdjacency:
	def __init__(self, nodes, sources, targets):
		self.nodes = nodes
		keys = np.sort(sources.numpy() * nodes + targets.numpy())
		capacity = 16
		while capacity < max(16, keys.size * 2):
			capacity *= 2
		self.data = np.full(capacity, -1, dtype=np.int64)
		self._spread(0, capacity, keys)

	def _spread(self, start, stop, keys):
		self.data[start:stop] = -1
		if keys.size:
			positions = start + (np.arange(keys.size) * (stop - start) // keys.size)
			self.data[positions] = keys

	def insert(self, row, neighbor):
		key = row * self.nodes + neighbor
		occupied = np.flatnonzero(self.data >= 0)
		position = np.searchsorted(self.data[occupied], key, side="right")
		anchor = int(occupied[position]) if position < occupied.size else self.data.size - 1
		size = 16
		while size <= self.data.size:
			start = anchor // size * size
			stop = start + size
			keys = self.data[start:stop]
			keys = keys[keys >= 0]
			if keys.size + 1 <= size * 0.75:
				self._spread(start, stop, np.insert(keys, np.searchsorted(keys, key, side="right"), key))
				return
			size *= 2
		keys = np.sort(np.append(self.data[self.data >= 0], key))
		self.data = np.full(self.data.size * 2, -1, dtype=np.int64)
		self._spread(0, self.data.size, keys)

	def delete(self, row, neighbor):
		self.data[self.data == row * self.nodes + neighbor] = -1

	def row(self, node):
		keys = self.data[(self.data >= node * self.nodes) & (self.data < (node + 1) * self.nodes)]
		return torch.from_numpy(keys % self.nodes)


class PMATopology:
	"""Graph facade consumed by affected analysis and normal worker task planning."""

	device = torch.device("cpu")
	metadata: dict[str, object] = {}

	def __init__(self, graph):
		if graph.device.type != "cpu" or not graph.is_homogeneous or "nrt" in graph.edata:
			raise ValueError("PMA topology requires a homogeneous CPU graph without edge attributes")
		self.nodes = graph.num_nodes()
		u, v = graph.edges()
		self.incoming = PackedAdjacency(self.nodes, v, u)
		self.outgoing = PackedAdjacency(self.nodes, u, v)
		self.edata = {}

	def num_nodes(self):
		return self.nodes

	def num_edges(self):
		return int((self.incoming.data >= 0).sum())

	def _validate(self, nodes):
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("PMA node IDs must be CPU int64 vectors")
		if nodes.numel() and (nodes.min() < 0 or nodes.max() >= self.nodes):
			raise ValueError("PMA node ID out of range")

	def _degrees(self, adjacency, nodes):
		nodes = torch.arange(self.nodes) if nodes is None else nodes
		self._validate(nodes)
		return torch.tensor([adjacency.row(int(node)).numel() for node in nodes], dtype=torch.int64)

	def in_degrees(self, nodes=None):
		return self._degrees(self.incoming, nodes)

	def out_degrees(self, nodes=None):
		return self._degrees(self.outgoing, nodes)

	def out_edges(self, nodes):
		self._validate(nodes)
		rows = [self.outgoing.row(int(node)) for node in nodes]
		return torch.repeat_interleave(nodes, torch.tensor([row.numel() for row in rows])), torch.cat(
			rows
		) if rows else nodes[:0]

	def has_edges_between(self, sources, targets):
		self._validate(sources)
		self._validate(targets)
		if sources.shape != targets.shape:
			raise ValueError("Edge endpoint lengths differ")
		return torch.tensor(
			[bool((self.outgoing.row(int(u)) == v).any()) for u, v in zip(sources, targets, strict=True)],
			dtype=torch.bool,
		)

	def add_edges(self, sources, targets):
		self.has_edges_between(sources, targets)
		for source, target in zip(sources.tolist(), targets.tolist(), strict=True):
			self.outgoing.insert(source, target)
			self.incoming.insert(target, source)

	def delete_pairs(self, sources, targets):
		self.has_edges_between(sources, targets)
		for source, target in zip(sources.tolist(), targets.tolist(), strict=True):
			self.outgoing.delete(source, target)
			self.incoming.delete(target, source)

	def compile_block(self, targets):
		import dgl

		self._validate(targets)
		rows = [self.incoming.row(int(node)) for node in targets]
		neighbors = torch.cat(rows) if rows else targets[:0]
		extra = neighbors.unique(sorted=True)
		extra = extra[~torch.isin(extra, targets)]
		sources = torch.cat([targets, extra])
		lookup = {int(node): index for index, node in enumerate(sources)}
		u = torch.tensor([lookup[int(node)] for node in neighbors], dtype=torch.int64)
		v = torch.repeat_interleave(torch.arange(targets.numel()), torch.tensor([row.numel() for row in rows]))
		block = dgl.create_block((u, v), num_src_nodes=sources.numel(), num_dst_nodes=targets.numel())
		block.srcdata[dgl.NID], block.dstdata[dgl.NID] = sources, targets
		return block

	def close(self):
		pass

	def create_formats_(self):
		pass
