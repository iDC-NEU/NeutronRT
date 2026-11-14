from unittest.mock import Mock

import pytest
import torch

from tests.graph.test_serving import _current_reference, _setup


def test_resume_does_not_repeat_edge_mutation_or_completion_credit():
	service, plane, storage, _ = _setup("sage")
	try:
		apply = service.apply
		service.apply = Mock(side_effect=RuntimeError("compute failure"))
		with pytest.raises(RuntimeError, match="compute failure"):
			service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		assert service.graph.num_edges() == 8
		assert service.catalog.version == 1
		assert service.counters.updates == 0
		with pytest.raises(RuntimeError, match="incomplete"):
			service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		assert service.graph.num_edges() == 8
		service.apply = apply
		result = service.resume_update(1)
		assert result.version == 1
		assert service.resume_update(1) is result
		assert service.graph.num_edges() == 8
		assert service.counters.updates == service.counters.input_changes == 1
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		plane.close()


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
def test_recover_owner_then_resume_an_interrupted_update(name, devices):
	service, plane, storage, _ = _setup(name, devices=devices)
	try:
		service.ensure(torch.arange(6))
		original = service.apply
		failed = []

		def fail_last_layer(worker, task, inputs):
			if task.layer == 2:
				failed.append(worker.owner)
				raise RuntimeError("worker failed during update")
			return original(worker, task, inputs)

		service.apply = fail_last_layer
		with pytest.raises(RuntimeError, match="worker failed"):
			service.insert_edges(torch.tensor([2]), torch.tensor([0]))
		assert service.graph.num_edges() == 8
		service.apply = original
		service.recover(failed[0])
		service.resume_update(1)
		assert service.graph.num_edges() == 8
		assert service.counters.updates == service.counters.input_changes == 1
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		plane.close()


def test_planning_failure_preserves_applied_input_and_has_a_conservative_resume():
	service, plane, storage, _ = _setup("sage")
	try:
		service.affected.analyze = Mock(side_effect=MemoryError("planning budget"))
		values = service.features[[1]] + 0.5
		with pytest.raises(MemoryError, match="planning budget"):
			service.change_features(torch.tensor([1]), values)
		torch.testing.assert_close(service.features[[1]], values)
		assert service.catalog.pending_layers == ()
		service.resume_update(1)
		assert service.counters.input_changes == 1
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
		with pytest.raises(ValueError, match="matching"):
			service.resume_update(2)
	finally:
		plane.close()


def test_failed_dispatch_releases_its_unpublished_catalog_reservation(monkeypatch):
	service, plane, _, _ = _setup("sage")
	try:
		reserved = []

		def reject(key, _owner):
			reserved.append(key)
			raise RuntimeError("dispatch rejected")

		monkeypatch.setattr(plane.directory, "begin", reject)
		with pytest.raises(RuntimeError, match="dispatch rejected"):
			service.ensure(torch.tensor([1]))
		assert reserved
		for key in reserved:
			with pytest.raises(KeyError):
				service.catalog.cancel(key)
	finally:
		plane.close()
