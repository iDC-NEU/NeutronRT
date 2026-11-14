"""Matched workloads and paired comparisons, independent of sockets and CUDA."""

import hashlib
import json
import math
import random
import statistics
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch import Tensor

from lib.runtime.config import RuntimeFeatures


def source_fingerprint(snapshot: Optional[Path] = None) -> str:
	"""Identify the source tree used at launch without exporting local paths."""
	root = Path(__file__).resolve().parents[2]
	paths = [root / "pyproject.toml", root / "benchmarks/sources.json"]
	for directory in (root / "lib", root / "scripts", root / "benchmarks"):
		paths.extend(path for path in directory.rglob("*") if path.suffix in (".py", ".cpp", ".proto"))
	digest = hashlib.sha256()
	with ZipFile(snapshot, "x", ZIP_DEFLATED) if snapshot is not None else nullcontext() as archive:
		for path in sorted(paths):
			content = path.read_bytes()
			name = path.relative_to(root).as_posix()
			digest.update(name.encode() + b"\0")
			digest.update(len(content).to_bytes(8, "little"))
			digest.update(content)
			if archive is not None:
				archive.writestr(name, content)
	return digest.hexdigest()


def inference_identity(graph, features, spec, *, model=None) -> str:
	"""Hash graph, feature, model and edge-attribute content outside measurement."""
	digest = hashlib.sha256()
	configuration = asdict(spec)
	# File locations and initialization seeds do not change already-built weights.
	configuration.pop("checkpoint")
	configuration.pop("seed")
	digest.update(json.dumps([graph.num_nodes(), graph.num_edges(), configuration], sort_keys=True).encode())
	model = spec.build() if model is None else model
	for value in (*graph.adj_tensors("csc")[:2], features, *model.state_dict().values()):
		for chunk in tensor_byte_chunks(value):
			digest.update(chunk)
	for key in sorted(graph.edata):
		digest.update(key.encode())
		for chunk in tensor_byte_chunks(graph.edata[key]):
			digest.update(chunk)
	return digest.hexdigest()


@dataclass(frozen=True)
class ProfileCase:
	name: str
	devices: Tuple[str, ...] = ("cpu",)
	batch_size: int = 8192
	interface: str = "sync"
	max_inflight: int = 4
	query_every: int = 0
	native_threads: int = 4
	max_stale_versions: int = 2
	query_staleness: int = 0
	idle_flush_seconds: float = 0.005
	switches: RuntimeFeatures = field(default_factory=RuntimeFeatures.baseline)
	reference: Optional[str] = None
	storage_partitions: int = 1
	delta_max_steps: int = 32
	delta_anchor_bytes: int = 256 * 1024**2

	def __post_init__(self) -> None:
		if (
			not isinstance(self.name, str)
			or not self.name
			or any(
				character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
				for character in self.name
			)
		):
			raise ValueError("Case names must be nonempty letters, digits, underscores, or hyphens")
		if not self.devices or any(not isinstance(value, str) for value in self.devices):
			raise ValueError("Each case needs explicit worker devices")
		devices = tuple(torch.device(value) for value in self.devices)
		if any(
			device.type not in ("cpu", "cuda") or device.type == "cuda" and device.index is None for device in devices
		):
			raise ValueError("Devices must be CPU or explicit CUDA indices")
		if len({device for device in devices if device.type == "cuda"}) != sum(
			device.type == "cuda" for device in devices
		):
			raise ValueError("One case cannot oversubscribe a CUDA device with several workers")
		for name in (
			"batch_size",
			"max_inflight",
			"query_every",
			"native_threads",
			"storage_partitions",
			"max_stale_versions",
			"query_staleness",
			"delta_max_steps",
			"delta_anchor_bytes",
		):
			value = getattr(self, name)
			if type(value) is not int or value < (
				0 if name in ("query_every", "max_stale_versions", "query_staleness", "delta_anchor_bytes") else 1
			):
				raise ValueError(f"Invalid case {name}")
		if self.interface not in ("sync", "async"):
			raise ValueError("Interface must be sync or async")
		if self.reference == self.name:
			raise ValueError("A comparison needs a different reference case")
		if self.reference is not None and not isinstance(self.reference, str):
			raise ValueError("A reference must name another case")
		if not isinstance(self.switches, RuntimeFeatures):
			raise ValueError("A case needs validated runtime switches")
		if self.storage_partitions > 1 and not self.switches.distributed_storage:
			raise ValueError("Multiple storage partitions require distributed_storage=true")
		if (
			type(self.idle_flush_seconds) not in (int, float)
			or not math.isfinite(self.idle_flush_seconds)
			or self.idle_flush_seconds <= 0
		):
			raise ValueError("Invalid idle flush interval")
		if self.query_staleness and (
			not self.switches.bounded_staleness or self.query_staleness > self.max_stale_versions
		):
			raise ValueError("Query staleness exceeds this case's enabled shared budget")

	@classmethod
	def from_dict(cls, value: Mapping[str, object]) -> "ProfileCase":
		values: dict[str, Any] = dict(value)
		if "switches" in values:
			if not isinstance(values["switches"], dict):
				raise ValueError("Case switches must be a JSON object")
			values["switches"] = RuntimeFeatures.from_dict(
				{**RuntimeFeatures.baseline().as_dict(), **values["switches"]}
			)
		if "devices" in values:
			if not isinstance(values["devices"], list):
				raise ValueError("Case devices must be a JSON list")
			values["devices"] = tuple(values["devices"])
		try:
			return cls(**values)
		except TypeError as error:
			raise ValueError("Unknown or missing case fields") from error

	@property
	def measured_switches(self) -> RuntimeFeatures:
		return replace(self.switches, profiling=True)


def read_cases(values: Sequence[Mapping[str, object]]) -> Tuple[ProfileCase, ...]:
	cases = tuple(ProfileCase.from_dict(value) for value in values)
	if not cases or len({case.name for case in cases}) != len(cases):
		raise ValueError("Cases must have unique names")
	names = {case.name for case in cases}
	if any(case.reference is not None and case.reference not in names for case in cases):
		raise ValueError("A reference case is absent")
	return cases


@dataclass(frozen=True)
class TraceUpdate:
	kind: str
	first: Tensor
	second: Tensor


def make_trace(
	features: Tensor,
	*,
	updates: int,
	changes: int,
	seed: int = 71,
	pool_size: int = 128,
	mixed: bool = False,
	max_bytes: int = 64 * 1024**2,
) -> Tuple[TraceUpdate, ...]:
	"""Absolute feature updates borrow no mutable server state; snapshot only a pool.

	A mixed trace pairs insertion/deletion requests. Existing parallel edges follow
	service deletion semantics and are checked by the independent full oracle.
	"""
	if features.device.type != "cpu" or features.ndim != 2 or features.dtype != torch.float32:
		raise ValueError("Trace source features must be a CPU float32 matrix")
	nodes, width = features.shape
	if min(updates, changes, nodes) < 1 or not changes <= pool_size <= nodes:
		raise ValueError("Invalid update count, changes, or hotspot pool size")
	if updates * changes * (width * 4 + 8) + pool_size * width * 4 > max_bytes:
		raise ValueError("Materialized workload exceeds its trace memory budget")
	generator = torch.Generator().manual_seed(seed)
	pool = torch.empty(0, dtype=torch.int64)
	if pool_size > nodes // 4:
		pool = torch.randperm(nodes, generator=generator)[:pool_size]
	else:
		while pool.numel() < pool_size:
			pool = torch.cat([pool, torch.randint(nodes, (2 * pool_size,), generator=generator)]).unique()
		pool = pool[torch.randperm(pool.numel(), generator=generator)[:pool_size]]
	base = features.index_select(0, pool)
	trace = []
	inserted = None
	for index in range(updates):
		selected = torch.randperm(pool_size, generator=generator)[:changes]
		first = pool[selected]
		if mixed and index % 3 == 1:
			second = pool[torch.randperm(pool_size, generator=generator)[:changes]]
			inserted = first, second
			trace.append(TraceUpdate("insert", first, second))
		elif mixed and index % 3 == 2:
			assert inserted is not None
			trace.append(TraceUpdate("delete", *inserted))
		else:
			second = base[selected] + torch.randn(changes, width, generator=generator) * 0.1
			trace.append(TraceUpdate("features", first, second))
	return tuple(trace)


def tensor_byte_chunks(value: Tensor, *, scratch_bytes: int = 1024**2) -> Iterator[memoryview]:
	"""Borrow contiguous CPU data; stream strided data in logical C order.

	Fingerprinting happens outside measurement. Never materialize a contiguous
	copy of the entire graph/feature tensor just to compute its identity.
	"""
	if value.device.type != "cpu" or scratch_bytes < value.element_size():
		raise ValueError("Fingerprinting requires CPU tensors and at least one element of scratch")
	if not value.numel():
		return
	array = value.detach().numpy()
	if array.flags.c_contiguous:
		yield memoryview(array).cast("B")
		return
	# NumPy 2.3's stub misspells the supported "contig" operand flag as "config".
	with np.nditer(  # type: ignore[call-overload]
		array,
		flags=["external_loop", "buffered", "zerosize_ok"],
		op_flags=["readonly", "contig"],
		order="C",
		buffersize=max(1, scratch_bytes // array.itemsize),
	) as chunks:
		for chunk in chunks:
			yield memoryview(chunk).cast("B")


def trace_digest(trace: Sequence[TraceUpdate]) -> str:
	digest = hashlib.sha256()
	for update in trace:
		digest.update(update.kind.encode())
		for value in (update.first, update.second):
			digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
			for chunk in tensor_byte_chunks(value):
				digest.update(chunk)
	return digest.hexdigest()


def latency_summary(seconds: Sequence[float]) -> dict:
	if not seconds:
		return {"count": 0, "p50_seconds": None, "p95_seconds": None, "p99_seconds": None}
	values = sorted(seconds)
	return {
		"count": len(values),
		**{
			f"p{percentile}_seconds": values[max(0, math.ceil(len(values) * percentile / 100) - 1)]
			for percentile in (50, 95, 99)
		},
	}


def paired_comparison(reference: Sequence[dict], candidate: Sequence[dict], *, metric: str = "client_seconds") -> dict:
	"""Lower metric is better. Never mix trace passes or hide failed trials."""
	left, right = {run["repetition"]: run for run in reference}, {run["repetition"]: run for run in candidate}
	if len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys() or not left:
		raise ValueError("Comparison requires unique, identical repetition sets")
	ratios = []
	for repetition in sorted(left):
		baseline, trial = left[repetition], right[repetition]
		if (
			not baseline["valid"]
			or not trial["valid"]
			or baseline.get("trace_pass")
			or trial.get("trace_pass")
			or baseline["trace_digest"] != trial["trace_digest"]
			or baseline["identity"] != trial["identity"]
		):
			raise ValueError("Trials are invalid, traced, or do not share workload/model identity")
		if any(
			not isinstance(run[metric], (int, float)) or not math.isfinite(run[metric]) or run[metric] <= 0
			for run in (baseline, trial)
		):
			raise ValueError("Comparison metric must be finite and positive")
		ratios.append(baseline[metric] / trial[metric])
	rng = random.Random(71)
	bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
	low, high = bootstrap[49], bootstrap[1949]
	verdict = "inconclusive"
	if len(ratios) >= 5:
		verdict = "gain" if low > 1.02 else "regression" if high < 1 / 1.02 else "inconclusive"
	return {
		"metric": metric,
		"pairs": len(ratios),
		"speedup_samples": ratios,
		"median_speedup": statistics.median(ratios),
		"paired_bootstrap_95_percent": [low, high],
		"verdict": verdict,
		"minimum_pairs_for_verdict": 5,
		"noise_margin_fraction": 0.02,
	}


def sample_ids(population: int, count: int, generator: torch.Generator) -> torch.Tensor:
	"""Uniform IDs without replacement and O(count) scratch for sparse batches."""
	if not 0 <= count <= population:
		raise ValueError("Sample count must be within the population")
	if count > population // 4:
		return torch.randperm(population, generator=generator)[:count]
	selected = torch.empty(0, dtype=torch.int64)
	while selected.numel() < count:
		remaining = count - selected.numel()
		candidates = torch.randint(population, (max(32, 2 * remaining),), generator=generator).unique()
		candidates = candidates[~torch.isin(candidates, selected)]
		order = torch.randperm(candidates.numel(), generator=generator)
		selected = torch.cat([selected, candidates[order[:remaining]]])
	return selected
