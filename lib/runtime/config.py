"""Explicit optimization switches, independent of numerical/resource settings."""

from dataclasses import asdict, dataclass, fields
from typing import Iterable, Mapping, Tuple, cast


@dataclass(frozen=True)
class RuntimeFeatures:
	pipeline: bool = False
	coalescing: bool = True
	peer_fetch: bool = True
	gather_inputs: bool = False
	preemption: bool = True
	recovery: bool = True
	profiling: bool = False
	native_analysis: bool = False
	incremental_aggregates: bool = False
	bounded_staleness: bool = False
	worker_feature_cache: bool = False
	alltoall: bool = False
	elastic_groups: bool = False
	distributed_storage: bool = False
	concurrent_compute: bool = False
	retain_aggregates: bool = False
	aggregate_only: bool = False
	row_refresh: bool = False
	operator_incremental: bool = False
	pma_storage: bool = False
	mapped_host_reads: bool = False

	def __post_init__(self) -> None:
		for field in fields(self):
			if type(getattr(self, field.name)) is not bool:
				raise ValueError(f"Feature switch {field.name} must be a boolean")

	@classmethod
	def baseline(cls) -> "RuntimeFeatures":
		"""An exact baseline with optional optimization/scheduling switches off."""
		return cls(**{field.name: False for field in fields(cls)})

	@classmethod
	def from_dict(cls, values: Mapping[str, object]) -> "RuntimeFeatures":
		unknown = set(values) - {field.name for field in fields(cls)}
		if unknown:
			raise ValueError(f"Unknown feature switches: {sorted(unknown)}")
		if any(type(value) is not bool for value in values.values()):
			raise ValueError("Feature switches must be JSON booleans, not strings or numbers")
		return cls(**{name: cast(bool, value) for name, value in values.items()})

	def as_dict(self) -> dict:
		return asdict(self)

	def require_supported(self, supported: Iterable[str]) -> None:
		missing = set(self.enabled) - set(supported)
		if missing:
			raise NotImplementedError(f"Feature implementations are not ready: {sorted(missing)}")

	@property
	def enabled(self) -> Tuple[str, ...]:
		return tuple(field.name for field in fields(self) if getattr(self, field.name))

	@property
	def mask(self) -> int:
		"""Compact startup identity; not a substitute for per-protocol wire versions."""
		return sum(1 << index for index, field in enumerate(fields(self)) if getattr(self, field.name))
