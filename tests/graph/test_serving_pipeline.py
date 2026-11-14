from threading import Event, get_ident
from unittest.mock import Mock

import pytest
import torch

from tests.graph.test_serving import _current_reference, _setup


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize(
	"devices",
	[
		("cpu", "cpu"),
		pytest.param(
			("cuda:0", "cuda:1"),
			marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs"),
		),
	],
)
def test_serial_and_pipeline_execute_equal_work_and_match_full_updates(name, devices):
	serial, serial_plane, serial_storage, _ = _setup(name, devices=devices, batch_size=2)
	pipeline, pipeline_plane, pipeline_storage, _ = _setup(name, devices=devices, batch_size=2)
	pipeline.pipeline = True
	try:
		for service in (serial, pipeline):
			service.ensure(torch.arange(6))
		for operation in range(3):
			results = []
			for service, storage in ((serial, serial_storage), (pipeline, pipeline_storage)):
				if operation == 0:
					result = service.change_features(torch.tensor([1, 5]), service.features[[1, 5]] + 0.75)
				elif operation == 1:
					result = service.insert_edges(torch.tensor([2, 4]), torch.tensor([0, 5]))
				else:
					result = service.delete_edges(torch.tensor([1]), torch.tensor([2]))
				results.append(result)
				with service.query(torch.arange(6), storage.owner) as values:
					torch.testing.assert_close(values, _current_reference(service), atol=2e-5, rtol=2e-5)
			assert results[0].affected == results[1].affected
			assert results[0].tasks == results[1].tasks
			assert not results[0].lookahead and results[1].lookahead
			assert serial.counters.rows == pipeline.counters.rows
			assert results[1].compute_seconds > 0
			assert results[1].preparation_seconds > 0
	finally:
		serial_plane.close()
		pipeline_plane.close()


def test_storage_prepares_next_layer_while_current_compute_is_running():
	service, plane, storage, _ = _setup("sage", batch_size=2)
	service.pipeline = True
	started, prepared = Event(), Event()
	coordinator = get_ident()
	apply, prepare = service.apply, service._prepare_layer

	def gated_apply(worker, task, inputs):
		assert get_ident() != coordinator
		if task.layer == 1:
			started.set()
			assert prepared.wait(10), "Next-layer preparation did not overlap current computation"
		return apply(worker, task, inputs)

	def observed_prepare(affected):
		assert get_ident() == coordinator
		if affected.layer == 2:
			assert started.wait(10)
		item = prepare(affected)
		if affected.layer == 2:
			prepared.set()
		return item

	service.apply, service._prepare_layer = gated_apply, observed_prepare
	try:
		service.change_features(torch.tensor([1]), service.features[[1]] + 1)
		service.apply = apply
		with service.query(torch.arange(6), storage.owner) as values:
			torch.testing.assert_close(values, _current_reference(service))
	finally:
		plane.close()


@pytest.mark.parametrize("failure", ["prepare", "compute"])
def test_pipeline_failure_drains_and_resumes_without_repeating_root_mutation(failure):
	service, plane, storage, _ = _setup("gat", batch_size=2)
	service.pipeline = True
	apply, prepare = service.apply, service._prepare_layer
	if failure == "prepare":

		def reject_next(affected):
			if affected.layer == 2:
				raise RuntimeError("pipeline injected failure")
			return prepare(affected)

		service._prepare_layer = reject_next
	else:
		service.apply = Mock(side_effect=RuntimeError("pipeline injected failure"))
	try:
		with pytest.raises(RuntimeError, match="pipeline injected"):
			service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		assert service.catalog.pending_layers == ()
		assert service.counters.updates == 0
		assert service.graph.num_edges() == 8
		service.apply, service._prepare_layer = apply, prepare
		service.resume_update(1)
		assert service.graph.num_edges() == 8
		assert service.counters.updates == service.counters.input_changes == 1
		with service.query(torch.arange(6), storage.owner) as values:
			torch.testing.assert_close(values, _current_reference(service))
	finally:
		plane.close()


def test_prefetch_budget_keeps_large_blocks_out_of_lookahead(monkeypatch):
	service, plane, _, _ = _setup("sage", batch_size=2)
	service.prefetch_edges = 0
	try:
		layers = list(service.affected.analyze(changed_features=torch.tensor([1])))
		block = Mock(side_effect=AssertionError("Unexpected block allocation"))
		monkeypatch.setattr(service.affected, "block", block)
		assert service._prepare_layer(layers[0]).first is None
	finally:
		plane.close()
