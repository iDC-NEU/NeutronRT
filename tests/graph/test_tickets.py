import gc
import weakref
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest
import torch

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.tickets import SessionMismatch, TicketRegistry
from tests.graph.test_serving import _setup


def test_concurrent_retries_share_admission_and_timeout_cannot_cancel_the_update():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service)
	registry = TicketRegistry(admission, capacity=1)
	started, proceed = Event(), Event()
	apply = service.apply

	def blocked(*args):
		started.set()
		assert proceed.wait(10)
		return apply(*args)

	service.apply = blocked
	first, second = torch.tensor([2]), torch.tensor([0])
	edges = service.graph.num_edges()
	try:
		with ThreadPoolExecutor(max_workers=4) as callers:
			entries = list(
				callers.map(
					lambda _: registry.submit(UpdateKind.INSERT, first, second, request_id="one-update"), range(12)
				)
			)
		entry = entries[0]
		assert started.wait(5) and all(other is entry for other in entries)
		assert admission.counters.admitted == 1
		assert not entry.future.cancel()
		with pytest.raises(TimeoutError):
			entry.result(timeout=0)
		with pytest.raises(RuntimeError, match="running"):
			registry.release(entry.session, entry.number, entry.request_id)
		with pytest.raises(AdmissionFull):
			registry.submit(UpdateKind.INSERT, first, second, request_id="another-update")
		proceed.set()
		result = entry.result(timeout=10)
		assert result.version == 1 and service.graph.num_edges() == edges + 1
		assert registry.get(entry.session, entry.number, entry.request_id).result() is result
		registry.release(entry.session, entry.number, entry.request_id)
		with pytest.raises(KeyError):
			registry.get(entry.session, entry.number, entry.request_id)
	finally:
		proceed.set()
		admission.close()
		plane.close()


def test_reused_request_ids_validate_payload_and_session_before_another_mutation():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service)
	registry = TicketRegistry(admission)
	try:
		entry = registry.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]), request_id="update")
		assert entry.result(timeout=10).version == 1
		for kind, target in ((UpdateKind.DELETE, 0), (UpdateKind.INSERT, 1)):
			with pytest.raises(ValueError, match="different"):
				registry.submit(kind, torch.tensor([2]), torch.tensor([target]), request_id="update")
		with pytest.raises(SessionMismatch):
			registry.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]), request_id="x", session="old")
		with pytest.raises(ValueError, match="identity"):
			registry.get(entry.session, entry.number, "different-id")
		assert admission.counters.admitted == 1 and service.catalog.version == 1
	finally:
		admission.close()
		plane.close()


def test_pending_tickets_never_expire_and_completed_retention_frees_capacity():
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service)
	now = [0.0]
	registry = TicketRegistry(admission, capacity=1, retention=5, clock=lambda: now[0])
	started, proceed = Event(), Event()
	insert = service.insert_edges

	def blocked(*args):
		started.set()
		assert proceed.wait(10)
		return insert(*args)

	service.insert_edges = blocked
	try:
		entry = registry.submit(UpdateKind.INSERT, torch.tensor([2]), torch.tensor([0]), request_id="update")
		assert started.wait(5)
		now[0] = 100
		assert registry.get(entry.session, entry.number, entry.request_id) is entry
		proceed.set()
		entry.result(timeout=10)
		now[0] = 104
		assert registry.get(entry.session, entry.number, entry.request_id) is entry
		now[0] = 106
		with pytest.raises(KeyError):
			registry.get(entry.session, entry.number, entry.request_id)
		assert (
			registry.submit(UpdateKind.DELETE, torch.tensor([2]), torch.tensor([0]), request_id="next")
			.result(10)
			.version
			== 2
		)
	finally:
		proceed.set()
		admission.close()
		plane.close()


def test_failed_ticket_retains_only_failure_metadata_and_releases_submitted_storage(monkeypatch):
	service, plane, _, _ = _setup("sage")
	admission = UpdateAdmission(service)
	registry = TicketRegistry(admission)
	observed = []

	def failed(_kind, payload):
		observed.append(weakref.ref(payload[1]))
		raise ValueError("Malformed feature update")

	admission._execute = failed
	# A buffered test log handler would itself retain exc_info's failure frames.
	monkeypatch.setattr("lib.graph.tickets.logging.error", lambda *args, **kwargs: None)
	try:
		entry = registry.submit(UpdateKind.FEATURES, torch.tensor([2]), service.features[[2]], request_id="bad")
		admission.close()
		gc.collect()
		assert observed[0]() is None
		errors = []
		for _ in range(2):
			with pytest.raises(ValueError, match="Malformed") as caught:
				entry.result(timeout=0)
			errors.append(caught.value)
		assert errors[0] is not errors[1]
		assert admission.counters.failed == 1 and admission.counters.pending_bytes == 0
	finally:
		admission.close()
		plane.close()
