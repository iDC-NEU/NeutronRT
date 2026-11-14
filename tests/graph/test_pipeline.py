from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest

from lib.graph.pipeline import execute_layer_pipeline


@pytest.mark.parametrize("lookahead", [False, True])
def test_layer_preparation_overlaps_only_when_enabled_and_compute_dependencies_are_preserved(lookahead):
	started, next_prepared = Event(), Event()
	trace = []
	control = get_ident()

	def prepare(layer):
		assert get_ident() == control
		if layer == 2 and lookahead:
			assert started.wait(5)
		trace.append(("prepare", layer))
		if layer == 2:
			next_prepared.set()
		return layer

	def compute(layer):
		assert get_ident() != control
		trace.append(("compute", layer))
		if layer == 1:
			started.set()
			if lookahead:
				assert next_prepared.wait(5)
		else:
			assert ("done", layer - 1) in trace
		trace.append(("done", layer))

	with ThreadPoolExecutor(1) as executor:
		timing = execute_layer_pipeline(range(1, 4), prepare, compute, executor=executor, lookahead=lookahead)
	assert timing.layers == 3
	assert timing.prepare_seconds >= 0 and timing.compute_seconds >= 0 and timing.wait_seconds >= 0
	if lookahead:
		assert trace.index(("prepare", 2)) < trace.index(("done", 1))
	else:
		assert trace.index(("done", 1)) < trace.index(("prepare", 2))


def test_preparation_failure_drains_running_compute_before_returning():
	finished = Event()

	def prepare(layer):
		if layer == 2:
			raise ValueError("preparation failed")
		return layer

	def compute(_layer):
		finished.set()

	with ThreadPoolExecutor(1) as executor:
		with pytest.raises(ValueError, match="preparation failed"):
			execute_layer_pipeline([1, 2], prepare, compute, executor=executor, lookahead=True)
		assert finished.is_set()


def test_compute_failure_does_not_execute_prepared_next_layer():
	executed = []

	def compute(layer):
		executed.append(layer)
		raise RuntimeError("GPU failed")

	with ThreadPoolExecutor(1) as executor, pytest.raises(RuntimeError, match="GPU failed"):
		execute_layer_pipeline([1, 2], lambda layer: layer, compute, executor=executor, lookahead=True)
	assert executed == [1]


def test_empty_pipeline_and_invalid_preparation():
	with ThreadPoolExecutor(1) as executor:
		assert (
			execute_layer_pipeline([], lambda layer: layer, lambda _: None, executor=executor, lookahead=True).layers
			== 0
		)
		with pytest.raises(ValueError, match="concrete"):
			execute_layer_pipeline([1], lambda _: None, lambda _: None, executor=executor, lookahead=True)
