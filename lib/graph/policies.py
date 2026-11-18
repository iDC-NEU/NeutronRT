"""Small, explicit policy objects used by the serving runtime.

The policies in this module are deliberately independent of the authoritative
embedding directory.  Dispatch may observe certified locations, but it never
changes ownership.  The feature-row cache is a derived worker-local copy and
is keyed by graph version so it cannot satisfy a request with an older row.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
from typing import Dict, Generic, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar

import torch
from torch import Tensor

from lib.graph.ownership import Owner


class OwnedWorker(Protocol):
	@property
	def owner(self) -> Owner: ...


Worker = TypeVar("Worker", bound=OwnedWorker)


@dataclass(frozen=True)
class DispatchChoice(Generic[Worker]):
	worker: Worker
	reserved_cost: int


class DispatchPolicy:
	"""Choose workers using round-robin or certified locality plus load.

	A task's integer cost is ``max(1, output_rows + compiled_edges)``.  The
	load ledger stores the sum of those units for tasks selected but not yet
	released.  Locality is the fraction of required input rows whose certified
	published location is on a candidate worker.  The locality/load score is
	``locality_weight * locality - load_weight * projected_load / cost``;
	ties use worker rank then generation.
	"""

	MODES = ("round_robin", "locality_load")

	def __init__(self, mode: str = "round_robin", *, locality_weight: float = 1.0, load_weight: float = 1.0) -> None:
		mode = mode.replace("-", "_")
		if mode not in self.MODES:
			raise ValueError(f"Unsupported dispatch policy: {mode!r}")
		if (
			not isfinite(locality_weight)
			or not isfinite(load_weight)
			or locality_weight < 0
			or load_weight < 0
			or locality_weight == 0
			and load_weight == 0
		):
			raise ValueError("Dispatch weights must be finite, nonnegative, and not both zero")
		self.mode = mode
		self.locality_weight = float(locality_weight)
		self.load_weight = float(load_weight)
		self._next = 0
		self._loads: Dict[Owner, int] = {}
		self.choices = 0
		self.locality_choices = 0
		self.serial_fallbacks = 0

	@property
	def loads(self) -> Mapping[Owner, int]:
		return dict(self._loads)

	def choose(
		self,
		workers: Sequence[Worker],
		*,
		locality_rows: Optional[Mapping[Owner, int]] = None,
		total_rows: int = 0,
		task_cost: int = 1,
	) -> DispatchChoice[Worker]:
		if not workers:
			raise RuntimeError("No live compute endpoints")
		if type(task_cost) is not int or task_cost < 1:
			raise ValueError("Dispatch task cost must be a positive integer")
		if self.mode == "round_robin":
			worker = workers[self._next % len(workers)]
			self._next += 1
		else:
			rows = locality_rows or {}
			if type(total_rows) is not int or total_rows < 0:
				raise ValueError("Dispatch locality row count must be a nonnegative integer")
			denominator = max(1, total_rows)
			cost = float(task_cost)

			def score(candidate: Worker) -> tuple[float, int, int]:
				value = self.locality_weight * rows.get(candidate.owner, 0) / denominator
				value -= self.load_weight * (self._loads.get(candidate.owner, 0) + task_cost) / cost
				return -value, candidate.owner.rank, candidate.owner.generation

			worker = min(workers, key=score)
			self.locality_choices += 1
			if len(workers) == 1:
				self.serial_fallbacks += 1
		self._loads[worker.owner] = self._loads.get(worker.owner, 0) + task_cost
		self.choices += 1
		return DispatchChoice(worker, task_cost)

	def adjust(self, owner: Owner, reserved_cost: int, actual_cost: int) -> None:
		if type(reserved_cost) is not int or type(actual_cost) is not int or min(reserved_cost, actual_cost) < 1:
			raise ValueError("Dispatch costs must be positive integers")
		current = self._loads.get(owner, 0)
		if current < reserved_cost:
			raise RuntimeError("Dispatch load ledger underflow")
		updated = current - reserved_cost + actual_cost
		if updated < 0:
			raise RuntimeError("Dispatch load ledger underflow")
		self._loads[owner] = updated

	def release(self, owner: Owner, reserved_cost: int) -> None:
		if type(reserved_cost) is not int or reserved_cost < 1:
			raise ValueError("Dispatch release cost must be a positive integer")
		current = self._loads.get(owner, 0)
		if current < reserved_cost:
			raise RuntimeError("Dispatch load ledger underflow")
		if current == reserved_cost:
			self._loads.pop(owner, None)
		else:
			self._loads[owner] = current - reserved_cost

	def profile(self) -> dict:
		return {
			"mode": self.mode,
			"locality_weight": self.locality_weight,
			"load_weight": self.load_weight,
			"cost_units": "output_rows_plus_compiled_edges",
			"locality_units": "certified_input_rows_fraction",
			"choices": self.choices,
			"locality_choices": self.locality_choices,
			"serial_fallbacks": self.serial_fallbacks,
			"outstanding_cost_units": {
				f"{owner.rank}:g{owner.generation}": value
				for owner, value in sorted(self._loads.items(), key=lambda item: (item[0].rank, item[0].generation))
			},
		}


@dataclass(frozen=True)
class DeferredPolicyConfig:
	"""Validated deferred-publication policy identity."""

	mode: str = "version_window"
	window_seconds: Optional[float] = None
	hotspot_degree_threshold: Optional[int] = None
	hotspot_max_changes: Optional[int] = None

	def __post_init__(self) -> None:
		mode = self.mode.replace("-", "_")
		object.__setattr__(self, "mode", mode)
		if mode not in ("version_window", "fixed_time", "hotspot"):
			raise ValueError(f"Unsupported deferred policy: {self.mode!r}")
		if mode == "version_window":
			if (
				self.window_seconds is not None
				or self.hotspot_degree_threshold is not None
				or self.hotspot_max_changes is not None
			):
				raise ValueError("Version-window policy does not accept fixed-time or hotspot options")
			return
		if self.window_seconds is None or not isfinite(self.window_seconds) or self.window_seconds <= 0:
			raise ValueError("Fixed-time and hotspot policies require a positive finite window")
		if mode == "fixed_time" and (self.hotspot_degree_threshold is not None or self.hotspot_max_changes is not None):
			raise ValueError("Fixed-time policy does not accept hotspot options")
		if mode == "hotspot":
			if type(self.hotspot_degree_threshold) is not int or self.hotspot_degree_threshold < 0:
				raise ValueError("Hotspot policy requires a nonnegative degree threshold")
			if type(self.hotspot_max_changes) is not int or self.hotspot_max_changes < 1:
				raise ValueError("Hotspot policy requires a positive maximum change count")

	def as_dict(self) -> dict:
		return {
			"mode": self.mode,
			"window_seconds": self.window_seconds,
			"hotspot_degree_threshold": self.hotspot_degree_threshold,
			"hotspot_max_changes": self.hotspot_max_changes,
			"version_bound": "max_versions",
		}


class FeatureRowCache:
	"""Logical byte-bounded LRU for versioned feature rows.

	The budget is an accounting budget for tensor payload bytes plus a fixed
	per-row metadata charge. It excludes Python/container allocator overhead and
	does not claim a physical RSS or device-memory guarantee.
	"""

	METADATA_BYTES_PER_ROW = 24
	ACCOUNTING_SCOPE = "tensor_payload_plus_logical_metadata"
	PHYSICAL_OVERHEAD_INCLUDED = False

	def __init__(self, max_bytes: int) -> None:
		if type(max_bytes) is not int or max_bytes < 0:
			raise ValueError("Feature-row cache budget must be a nonnegative integer")
		self.max_bytes = max_bytes
		self._rows: OrderedDict[Tuple[int, int], Tensor] = OrderedDict()
		self._version: Optional[int] = None
		self.data_bytes = 0
		self.hits = self.misses = self.hit_bytes = self.miss_bytes = 0
		self.evictions = self.invalidations = self.oversized_rows = 0
		self.invalidated_bytes = 0
		self._row_shape: Optional[Tuple[int, ...]] = None
		self._dtype: Optional[torch.dtype] = None
		self._device: Optional[torch.device] = None

	@property
	def metadata_bytes(self) -> int:
		return len(self._rows) * self.METADATA_BYTES_PER_ROW

	@property
	def bytes_used(self) -> int:
		return self.data_bytes + self.metadata_bytes

	def _remove(self, key: Tuple[int, int]) -> None:
		value = self._rows.pop(key)
		self.data_bytes -= value.numel() * value.element_size()

	def invalidate(self) -> None:
		self.invalidations += len(self._rows)
		self.invalidated_bytes += self.bytes_used
		self._rows.clear()
		self.data_bytes = 0

	def _insert(self, key: Tuple[int, int], value: Tensor) -> None:
		row_bytes = value.numel() * value.element_size()
		entry_bytes = row_bytes + self.METADATA_BYTES_PER_ROW
		if entry_bytes > self.max_bytes:
			self.oversized_rows += 1
			return
		if key in self._rows:
			self._remove(key)
		while self._rows and self.bytes_used + entry_bytes > self.max_bytes:
			old_key, _ = next(iter(self._rows.items()))
			self._remove(old_key)
			self.evictions += 1
		self._rows[key] = value.detach().clone()
		self.data_bytes += row_bytes

	def resolve(self, version: int, nodes: Tensor, values: Tensor) -> Tensor:
		if type(version) is not int or version < 0:
			raise ValueError("Feature cache versions must be nonnegative integers")
		if nodes.device.type not in ("cpu", "cuda") or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Feature cache node IDs must be one-dimensional CPU or CUDA int64")
		if (
			values.device.type not in ("cpu", "cuda")
			or values.dtype != torch.float32
			or values.ndim != 2
			or values.shape[0] != nodes.numel()
		):
			raise ValueError("Feature cache values must be CPU or CUDA float32 rows aligned with node IDs")
		if self._version != version:
			if self._version is not None:
				self.invalidate()
			self._version = version
			self._row_shape = tuple(values.shape[1:])
			self._dtype = values.dtype
			self._device = values.device
		elif self._row_shape != tuple(values.shape[1:]) or self._dtype != values.dtype or self._device != values.device:
			raise ValueError("Feature cache row shape, dtype, and device cannot change within a version")
		result = torch.empty_like(values)
		row_bytes = values.shape[1] * values.element_size()
		for index, node in enumerate(nodes.detach().cpu().tolist()):
			key = (version, int(node))
			cached = self._rows.pop(key, None)
			if cached is not None:
				result[index].copy_(cached)
				self._rows[key] = cached
				self.hits += 1
				self.hit_bytes += row_bytes
				continue
			result[index].copy_(values[index])
			self.misses += 1
			self.miss_bytes += row_bytes
			self._insert(key, values[index])
		return result

	def profile(self) -> dict:
		return {
			"enabled": True,
			"budget_bytes": self.max_bytes,
			"accounting_scope": self.ACCOUNTING_SCOPE,
			"metadata_bytes_per_row": self.METADATA_BYTES_PER_ROW,
			"physical_overhead_included": self.PHYSICAL_OVERHEAD_INCLUDED,
			"bytes_used": self.bytes_used,
			"data_bytes": self.data_bytes,
			"metadata_bytes": self.metadata_bytes,
			"hits": self.hits,
			"misses": self.misses,
			"hit_bytes": self.hit_bytes,
			"miss_bytes": self.miss_bytes,
			"evictions": self.evictions,
			"invalidations": self.invalidations,
			"invalidated_bytes": self.invalidated_bytes,
			"oversized_rows": self.oversized_rows,
		}


class FallbackCounters:
	"""Per-layer sparse-delta fallback counts with explicit reason scope."""

	def __init__(self) -> None:
		self._counts: Dict[Tuple[int, str], int] = {}
		self._rows: Dict[Tuple[int, str], int] = {}

	def add(self, layer: int, reason: str, rows: int) -> None:
		if type(layer) is not int or layer < 1 or type(rows) is not int or rows < 0 or not reason:
			raise ValueError("Fallback counters require a positive layer, reason, and nonnegative row count")
		key = layer, reason
		self._counts[key] = self._counts.get(key, 0) + 1
		self._rows[key] = self._rows.get(key, 0) + rows

	def as_dict(self) -> dict:
		layers: Dict[str, dict] = {}
		for layer, reason in sorted(self._counts):
			layers.setdefault(f"layer{layer}", {})[reason] = {
				"tasks": self._counts[layer, reason],
				"rows": self._rows[layer, reason],
			}
		return {
			"scope": "one increment per full-fallback task or selected fallback row",
			"layers": layers,
		}
