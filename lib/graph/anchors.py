"""Compact old-row routing while catalog heads advance to a new graph version."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from lib.graph.catalog import PriorRows, RowSelection


@dataclass(frozen=True)
class AnchorSelection:
	covered: Tensor
	available: Tensor
	selections: Tuple[RowSelection, ...]


class AnchoredRows:
	"""Selected integer metadata only; caller retains the referenced state keys."""

	def __init__(self, snapshot: PriorRows) -> None:
		nodes, valid = snapshot.nodes, snapshot.available
		if (
			nodes.device.type != "cpu"
			or nodes.dtype != torch.int64
			or nodes.ndim != 1
			or not torch.equal(nodes, nodes.unique(sorted=True))
			or valid.device.type != "cpu"
			or valid.dtype != torch.bool
			or valid.shape != nodes.shape
		):
			raise ValueError("Old-row anchors require sorted unique CPU nodes and aligned validity")
		self.snapshot = snapshot
		self._groups = torch.full_like(nodes, -1)
		self._offsets = torch.zeros_like(nodes)
		positions = []
		for index, group in enumerate(snapshot.selections):
			if (
				group.key.layer != snapshot.layer
				or group.key.version > snapshot.version
				or group.positions.device.type != "cpu"
				or group.rows.device.type != "cpu"
				or group.positions.dtype != torch.int64
				or group.rows.dtype != torch.int64
				or group.positions.ndim != 1
				or group.rows.shape != group.positions.shape
				or group.positions.numel()
				and (group.positions.min() < 0 or group.positions.max() >= nodes.numel() or group.rows.min() < 0)
			):
				raise ValueError("Old-row routes must match their certified layer/version and row positions")
			positions.append(group.positions)
			self._groups[group.positions] = index
			self._offsets[group.positions] = group.rows
		covered = torch.cat(positions) if positions else torch.empty(0, dtype=torch.int64)
		if not torch.equal(covered.sort().values, valid.nonzero().flatten()):
			raise ValueError("Old-row routes must cover each available anchor exactly once")

	def lookup(self, nodes: Tensor) -> AnchorSelection:
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Old-row lookups require CPU int64 node IDs")
		if not self.snapshot.nodes.numel():
			return AnchorSelection(
				torch.zeros_like(nodes, dtype=torch.bool), torch.zeros_like(nodes, dtype=torch.bool), ()
			)
		indices = torch.searchsorted(self.snapshot.nodes, nodes)
		indices.clamp_max_(self.snapshot.nodes.numel() - 1)
		covered = self.snapshot.nodes[indices] == nodes
		available = covered & self.snapshot.available[indices]
		groups = self._groups[indices]
		selections = []
		for index in groups[available].unique(sorted=True).tolist():
			positions = (available & (groups == index)).nonzero().flatten()
			selections.append(
				RowSelection(self.snapshot.selections[index].key, positions, self._offsets[indices[positions]])
			)
		return AnchorSelection(covered, available, tuple(selections))
