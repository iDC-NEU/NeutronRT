"""Small, explicit hooks for counters exposed by the serving/runtime owner.

The profiler observes counters; it does not invent policy, cache, or fallback
events. Runtime code can expose scalar counters or nested mappings through the
same snapshot/diff contract without requiring a second runtime implementation.
"""

from collections.abc import Mapping
from numbers import Number
from typing import Any, cast

_CONFIG_KEYS = {
	"enabled",
	"status",
	"mode",
	"locality_weight",
	"load_weight",
	"cost_units",
	"locality_units",
	"version_bound",
	"window_seconds",
	"hotspot_degree_threshold",
	"hotspot_max_changes",
	"budget_bytes",
	"max_bytes",
	"metadata_bytes_per_row",
	"accounting_scope",
	"physical_overhead_included",
	"device",
	"host",
	"rank",
	"generation",
}
_GAUGE_KEYS = {
	"data_bytes",
	"metadata_bytes",
	"outstanding_cost_units",
	"loads",
	"resident_current_bytes",
	"history_bytes",
}


def counter_snapshot(source: Any) -> dict[str, Any]:
	"""Copy numeric scalar counters and nested mappings without live references."""
	values = source if isinstance(source, Mapping) else vars(source)
	return {str(name): _snapshot(value) for name, value in values.items() if _is_counter(value)}


def _is_counter(value: Any) -> bool:
	return isinstance(value, Mapping) or (isinstance(value, Number) and not isinstance(value, (bool, complex)))


def _snapshot(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(name): _snapshot(item) for name, item in value.items() if _is_counter(item)}
	if isinstance(value, Number) and not isinstance(value, (bool, complex)):
		return value
	raise TypeError(f"Unsupported counter value: {type(value).__name__}")


def counter_delta(
	before: Mapping[str, Any], after: Mapping[str, Any], *, include_new: bool = True
) -> tuple[dict[str, Any], list[str]]:
	"""Diff counters and report resets instead of emitting negative work.

	A reset is explicit in the returned path list; its value is reported as zero
	for the window. New numeric paths start from zero; callers that need legacy
	intersection-only behavior can pass ``include_new=False``.
	"""
	resets: list[str] = []

	def visit(left: Any, right: Any, path: str) -> Any:
		if isinstance(right, Mapping):
			left = left if isinstance(left, Mapping) else {}
			result = {}
			names = right.keys() if include_new else right.keys() & left.keys()
			for name in names:
				value = visit(left.get(name), right[name], f"{path}.{name}" if path else str(name))
				if value is not None:
					result[str(name)] = value
			return result
		if (
			include_new
			and isinstance(right, Number)
			and not isinstance(right, (bool, complex))
			and not (isinstance(left, Number) and not isinstance(left, (bool, complex)))
		):
			return right
		if (
			isinstance(left, Number)
			and isinstance(right, Number)
			and not isinstance(left, (bool, complex))
			and not isinstance(right, (bool, complex))
		):
			if cast(Any, right) < left:
				resets.append(path)
				return 0
			return cast(Any, right) - left
		return None

	result = visit(before, after, "")
	return result if isinstance(result, dict) else {}, resets


def grouped_counter_delta(counters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
	"""Group only counters actually exposed by the runtime owner."""
	groups: dict[str, dict[str, Any]] = {"policy": {}, "fallback": {}, "cache": {}}
	for name, value in counters.items():
		lower = name.lower()
		if "fallback" in lower or "refresh" in lower:
			groups["fallback"][name] = value
		elif "cache" in lower or "reuse" in lower or "retained" in lower:
			groups["cache"][name] = value
		elif "policy" in lower or "dispatch" in lower or "wave" in lower:
			groups["policy"][name] = value
	return {name: values for name, values in groups.items() if values}


def profile_snapshot(source: Any) -> Any:
	"""Copy a runtime profile while preserving configuration and gauges."""
	if isinstance(source, Mapping):
		return {str(name): profile_snapshot(value) for name, value in source.items()}
	if isinstance(source, (list, tuple)):
		return [profile_snapshot(value) for value in source]
	if source is None or isinstance(source, (str, bool, int, float)):
		return source
	return None


def profile_window(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
	"""Separate exposed policy configuration, gauges, and monotonic counters.

	Only fields present in the runtime snapshots are reported. Byte budgets and
	current logical occupancy are never treated as work counters.
	"""
	configuration: dict[str, Any] = {}
	gauges: dict[str, Any] = {}
	left_counters: dict[str, Any] = {}
	right_counters: dict[str, Any] = {}
	new_counter_paths: list[str] = []

	def visit(left: Any, right: Any, path: str) -> None:
		if isinstance(right, Mapping):
			left_map = left if isinstance(left, Mapping) else {}
			for name, value in right.items():
				key = str(name)
				child = f"{path}.{key}" if path else key
				if key in _CONFIG_KEYS:
					configuration[child] = profile_snapshot(value)
				elif key in _GAUGE_KEYS:
					gauges[child] = profile_snapshot(value)
				else:
					visit(left_map.get(name), value, child)
			return
		if isinstance(right, Number) and not isinstance(right, (bool, complex)):
			if isinstance(left, Number) and not isinstance(left, (bool, complex)):
				left_counters[path] = left
			else:
				left_counters[path] = 0
				new_counter_paths.append(path)
			right_counters[path] = right

	visit(before, after, "")
	deltas, resets = counter_delta(_unflatten(left_counters), _unflatten(right_counters), include_new=True)
	return {
		"before": profile_snapshot(before),
		"after": profile_snapshot(after),
		"configuration": configuration,
		"gauges": gauges,
		"counter_deltas": deltas,
		"new_counter_paths": new_counter_paths,
		"counter_resets": resets,
		"reset_semantics": "Decreased exposed counters are reported as zero and named in counter_resets.",
	}


def _unflatten(values: Mapping[str, Any]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for path, value in values.items():
		current = result
		parts = path.split(".")
		for part in parts[:-1]:
			current = current.setdefault(part, {})
		current[parts[-1]] = value
	return result
