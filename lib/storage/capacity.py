"""Capacity planning for file-backed graph storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lib.storage.base import BaseTopology

GIB = 1024**3


@dataclass(frozen=True)
class StorageNodeBudget:
	name: str
	capacity_bytes: int = 512 * GIB
	reserve_bytes: int = 64 * GIB

	def __post_init__(self) -> None:
		if not self.name or min(self.capacity_bytes, self.reserve_bytes) < 0:
			raise ValueError("Storage node names and byte budgets must be nonnegative")
		if self.reserve_bytes >= self.capacity_bytes:
			raise ValueError("Storage reserve must leave positive usable capacity")

	@property
	def usable_bytes(self) -> int:
		return self.capacity_bytes - self.reserve_bytes


@dataclass(frozen=True)
class StoragePartitionPlan:
	node: str
	partition: int
	start: int
	stop: int
	base_bytes: int
	usable_bytes: int

	@property
	def fits(self) -> bool:
		return self.base_bytes <= self.usable_bytes


def _row_bytes(base: BaseTopology, direction: str, start: int, stop: int) -> int:
	offsets_path, neighbors_path = base.files(direction)
	offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
	neighbors = np.load(neighbors_path, mmap_mode="r", allow_pickle=False)
	return int((stop - start) * offsets.dtype.itemsize + (offsets[stop] - offsets[start]) * neighbors.dtype.itemsize)


def partition_base_bytes(base: BaseTopology, start: int, stop: int) -> int:
	"""Return mapped adjacency bytes for an arbitrary vertex partition."""
	if not 0 <= start <= stop <= base.nodes:
		raise ValueError("Storage partition must be within the exported vertex range")
	return sum(_row_bytes(base, direction, start, stop) for direction in ("in", "out"))


def partition_plan(base: BaseTopology, budgets: Sequence[StorageNodeBudget]) -> tuple[StoragePartitionPlan, ...]:
	if not budgets:
		raise ValueError("At least one storage node is required")
	plans = []
	for partition, budget in enumerate(budgets):
		start = base.nodes * partition // len(budgets)
		stop = base.nodes * (partition + 1) // len(budgets)
		base_bytes = partition_base_bytes(base, start, stop)
		plans.append(StoragePartitionPlan(budget.name, partition, start, stop, base_bytes, budget.usable_bytes))
	return tuple(plans)


def require_capacity(base: BaseTopology, budgets: Sequence[StorageNodeBudget]) -> tuple[StoragePartitionPlan, ...]:
	plans = partition_plan(base, budgets)
	over = [plan for plan in plans if not plan.fits]
	if over:
		details = ", ".join(
			f"{item.node}: requires {item.base_bytes} bytes, usable {item.usable_bytes} bytes" for item in over
		)
		raise MemoryError(f"Graph storage exceeds the configured node budget: {details}")
	return plans


def manifest(base: BaseTopology, budgets: Sequence[StorageNodeBudget]) -> dict:
	plans = require_capacity(base, budgets)
	return {
		"nodes": [
			{
				"name": budget.name,
				"capacity_bytes": budget.capacity_bytes,
				"reserve_bytes": budget.reserve_bytes,
				"usable_bytes": budget.usable_bytes,
			}
			for budget in budgets
		],
		"partitions": [
			{
				"node": plan.node,
				"partition": plan.partition,
				"start": plan.start,
				"stop": plan.stop,
				"base_bytes": plan.base_bytes,
				"usable_bytes": plan.usable_bytes,
				"fits": plan.fits,
			}
			for plan in plans
		],
		"scope": "Mapped adjacency base bytes only; overlay and response budgets must be added separately",
	}
