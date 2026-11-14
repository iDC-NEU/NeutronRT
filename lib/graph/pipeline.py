"""Bounded CPU preparation/GPU execution overlap with explicit layer barriers."""

import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Optional, TypeVar

Work = TypeVar("Work")
Prepared = TypeVar("Prepared")


@dataclass
class PipelineTiming:
	prepare_seconds: float = 0.0
	compute_seconds: float = 0.0
	wait_seconds: float = 0.0
	layers: int = 0


def execute_layer_pipeline(
	work: Iterable[Work],
	prepare: Callable[[Work], Prepared],
	compute: Callable[[Prepared], None],
	*,
	executor: Executor,
	lookahead: bool,
) -> PipelineTiming:
	"""Prepare at most one following layer; execute it only after its predecessor."""
	iterator = iter(work)
	timing = PipelineTiming()

	def advance() -> Optional[Prepared]:
		start = perf_counter()
		try:
			try:
				item = next(iterator)
			except StopIteration:
				return None
			value = prepare(item)
			if value is None:
				raise ValueError("Layer preparation must return a concrete work item")
			return value
		finally:
			timing.prepare_seconds += perf_counter() - start

	def execute(value: Prepared) -> float:
		start = perf_counter()
		compute(value)
		return perf_counter() - start

	current = advance()
	while current is not None:
		future = executor.submit(execute, current)
		try:
			following = advance() if lookahead else None
		except BaseException:
			try:
				future.result()
			except BaseException as error:
				logging.warning("Compute also failed while draining a preparation failure: %s", error)
			raise
		start = perf_counter()
		timing.compute_seconds += future.result()
		timing.wait_seconds += perf_counter() - start
		timing.layers += 1
		if not lookahead:
			following = advance()
		current = following
	return timing
