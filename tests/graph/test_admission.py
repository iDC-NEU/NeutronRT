from concurrent.futures import TimeoutError
from threading import Event

import pytest
import torch

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.data_plane import copy_tensor
from tests.graph.test_serving import _current_reference, _setup


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
def test_async_admission_returns_before_compute_and_sync_waits_for_publication(devices):
	service, plane, storage, _ = _setup("sage", devices=devices)
	service.pipeline = True
	admission = UpdateAdmission(service, max_pending=2)
	started, proceed = Event(), Event()
	apply = service.apply

	def gated_apply(worker, task, inputs):
		started.set()
		assert proceed.wait(10)
		return apply(worker, task, inputs)

	service.apply = gated_apply
	try:
		ticket = admission.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]))
		assert started.wait(5)
		assert not ticket.future.done()
		assert not ticket.future.cancel()
		with pytest.raises(TimeoutError):
			ticket.result(timeout=0)
		assert admission.counters.completed == 0
		proceed.set()
		result = ticket.result(timeout=10)
		assert result.version == 1
		assert ticket.result() is result
		assert admission.sync(UpdateKind.DELETE, torch.tensor([2]), torch.tensor([0])).version == 2
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		proceed.set()
		admission.close()
		plane.close()
	assert admission.counters.admitted == admission.counters.completed == 2
	assert admission.counters.pending == admission.counters.pending_bytes == 0


def test_pending_limit_and_payload_snapshot_do_not_block_or_copy_the_graph():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service, max_pending=2, max_bytes=4096)
	started, proceed = Event(), Event()
	original = service.insert_edges

	def gated_insert(*args):
		started.set()
		assert proceed.wait(10)
		return original(*args)

	service.insert_edges = gated_insert
	try:
		first = admission.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]))
		assert started.wait(5)
		values = service.features[[1]] + 1
		expected = values.clone()
		second = admission.submit(UpdateKind.FEATURES, torch.tensor([1]), values)
		values.fill_(-99)
		with pytest.raises(AdmissionFull):
			admission.submit(UpdateKind.DELETE, torch.tensor([2]), torch.tensor([0]))
		assert admission.counters.pending == 2
		proceed.set()
		assert first.result(timeout=10).version == 1
		assert second.result(timeout=10).version == 2
		torch.testing.assert_close(service.features[[1]], expected)
	finally:
		proceed.set()
		admission.close()
		plane.close()


def test_borrowed_payload_uses_original_storage_and_charges_entire_views():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service, max_bytes=24)
	try:
		base = torch.arange(8)
		with pytest.raises(AdmissionFull):
			admission.submit(UpdateKind.INSERT, base[:1], base[1:2], borrow=True)
		assert admission.counters.admitted == 0
		# Snapshotting only the submitted IDs fits the same budget.
		assert admission.sync(UpdateKind.INSERT, base[:1], base[1:2]).version == 1
		admission.max_bytes = 4096
		values = service.features[[2]] + 1
		original = service.change_features
		seen = []

		def check_pointer(nodes, submitted):
			seen.append(submitted.data_ptr())
			return original(nodes, submitted)

		service.change_features = check_pointer
		admission.sync(UpdateKind.FEATURES, torch.tensor([2]), values, borrow=True)
		assert seen == [values.data_ptr()]
	finally:
		admission.close()
		plane.close()


def test_sync_completion_does_not_wait_for_storage_writeback():
	service, plane, storage, _ = _setup("gat")
	admission = UpdateAdmission(service)
	started, proceed = Event(), Event()
	blocked_pointer = None

	def blocked_copy(value, device):
		if value.data_ptr() == blocked_pointer:
			started.set()
			assert proceed.wait(10)
		return copy_tensor(value, device)

	try:
		admission.sync(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]))
		key = service.catalog.selections(2, torch.tensor([0]), 1)[0].key
		with plane.acquire(key) as value:
			blocked_pointer = value.data_ptr()
		plane._transfer = blocked_copy
		writeback = plane.writeback(key, storage.owner)
		assert started.wait(5)
		# Updating another version does not depend on the blocked old writeback.
		result = admission.sync(UpdateKind.FEATURES, torch.tensor([5]), service.features[[5]] + 0.5)
		assert result.version == 2
		assert not writeback.done()
	finally:
		proceed.set()
		admission.close()
		plane.close()


def test_errors_propagate_and_capacity_is_released():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service, max_pending=1)
	try:
		with pytest.raises(ValueError):
			admission.sync(UpdateKind.INSERT, torch.tensor([99]), torch.tensor([0]))
		admission.close()
		assert admission.counters.failed == 1
		assert admission.counters.pending == admission.counters.pending_bytes == 0
		with pytest.raises(RuntimeError, match="closed"):
			admission.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]))
	finally:
		admission.close()
		plane.close()


def test_admission_profile_records_monotonic_lifecycle_and_versions():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service)
	try:
		ticket = admission.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]))
		result = ticket.result(timeout=10)
		profile = admission.profile(ticket.sequence)
		assert profile is not None
		assert profile["state"] == "published"
		assert profile["input_version"] == result.input_version
		assert profile["publication_version"] == result.version
		assert profile["admitted_at_ns"] <= profile["execution_started_at_ns"]
		assert profile["execution_started_at_ns"] <= profile["publication_observed_at_ns"]
		assert profile["publication_observed_at_ns"] <= profile["completion_at_ns"]
		assert profile["queue_wait_ns"] >= 0
		assert profile["execution_ns"] >= 0
		assert profile["delayed_publication_wait_ns"] == 0
		assert profile["server_reply_prepared_at_ns"] is None
		assert profile["clock_scope"].startswith("server perf_counter_ns")
		observed = admission.profile(ticket.sequence, observed_at_ns=profile["completion_at_ns"] + 1)
		assert observed["server_reply_prepared_at_ns"] == profile["completion_at_ns"] + 1
	finally:
		admission.close()
		plane.close()


def test_deferred_admission_separates_execution_from_publication_wait():
	service, plane, _, _ = _setup("sage")
	from lib.graph.deferred import DeferredUpdates

	deferred = DeferredUpdates(service, max_versions=2, idle_seconds=60)
	admission = UpdateAdmission(service)
	try:
		entered_submit, release_submit = Event(), Event()
		original_submit = deferred.submit

		def gated_submit(*args, **kwargs):
			publication = original_submit(*args, **kwargs)
			entered_submit.set()
			assert release_submit.wait(10)
			return publication

		deferred.submit = gated_submit
		ticket = admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1)
		assert ticket.telemetry is not None
		assert entered_submit.wait(10)
		queued = admission.profile(ticket.sequence)
		assert queued["state"] == "executing"
		release_submit.set()
		deferred.flush()
		result = ticket.result(timeout=10)
		profile = admission.profile(ticket.sequence)
		assert profile["publication_version"] == result.version
		assert profile["staged_at_ns"] is not None
		assert profile["delayed_publication_wait_ns"] >= 0
		assert profile["execution_ns"] >= 0
		assert profile["publication_source"] == "deferred_future_callback"
	finally:
		admission.close()
		deferred.close()
		plane.close()
