from dataclasses import replace

import pytest
import torch

from lib.graph.admission import AdmissionFull, UpdateAdmission, UpdateKind
from lib.graph.deferred import DeferredUpdates
from lib.runtime.policies import DeferredPolicyConfig
from tests.graph.test_serving import _current_reference, _setup


class _FakeTimer:
	def __init__(self, seconds, callback, *, args):
		self.seconds, self.callback, self.args = seconds, callback, args
		self.daemon = False
		self.started = False
		self.cancelled = False

	def start(self):
		self.started = True

	def cancel(self):
		self.cancelled = True


def _initialized(name="sage", *, max_versions=10, idle_seconds=60, policy=None, clock=None, timers=None):
	service, plane, storage, _ = _setup(name, batch_size=2)
	service.ensure(torch.arange(6))
	timer_factory = None
	if timers is not None:

		def timer_factory(seconds, callback, args):
			timer = _FakeTimer(seconds, callback, args=args)
			timers.append(timer)
			return timer

	deferred = DeferredUpdates(
		service,
		max_versions=max_versions,
		idle_seconds=idle_seconds,
		policy=policy,
		**({"clock": clock, "timer_factory": timer_factory} if clock is not None else {}),
	)
	return service, plane, storage, deferred


def test_fixed_time_waiting_is_distinct_from_the_version_window_baseline():
	clock = [0.0]
	timers = []
	service, plane, _, deferred = _initialized(
		policy=DeferredPolicyConfig(mode="fixed_time", window_seconds=10), clock=lambda: clock[0], timers=timers
	)
	try:
		first = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		second = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 2)
		assert not first.done() and not second.done() and deferred.pending == 2
		assert len(timers) == 1 and timers[0].seconds == 10
		clock[0] = 5
		timers[0].callback(*timers[0].args)
		assert not first.done() and deferred.pending == 2
		assert len(timers) == 2 and timers[1].seconds == 5
		third = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 3)
		assert not first.done() and not third.done() and len(timers) == 2
		clock[0] = 10
		fourth = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 4)
		assert first.result().version == second.result().version == 3
		assert not fourth.done() and deferred.pending == 1
		assert deferred.profile()["mode"] == "fixed_time"
		deferred.flush()
		assert third.result().version == 3 and fourth.result().version == 4
	finally:
		deferred.close()
		plane.close()


def test_hotspot_policy_flushes_non_hot_changes_and_caps_hotspot_batches():
	service, plane, _, deferred = _initialized(
		policy=DeferredPolicyConfig(
			mode="hotspot", window_seconds=30, hotspot_degree_threshold=1, hotspot_max_changes=2
		),
		clock=lambda: 0.0,
		timers=[],
	)
	try:
		first = deferred.submit("features", torch.tensor([0]), service.features[[0]] + 1)
		second = deferred.submit("features", torch.tensor([0]), service.features[[0]] + 2)
		assert first.done() and second.done() and first.result().version == second.result().version == 2
		third = deferred.submit("features", torch.tensor([0]), service.features[[0]] + 3)
		fourth = deferred.submit("features", torch.tensor([5]), service.features[[5]] + 4)
		assert third.done() and fourth.done()
		assert third.result().version == 3 and fourth.result().version == 4
	finally:
		deferred.close()
		plane.close()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_hot_updates_merge_real_work_across_layers_and_complete_only_after_exact_publication(name, pipeline):
	service, plane, storage, deferred = _initialized(name)
	service.pipeline = pipeline
	baseline, baseline_plane, _, _ = _setup(name, batch_size=2)
	baseline.ensure(torch.arange(6))
	before, baseline_before = replace(service.counters), replace(baseline.counters)
	pointer = service.features.data_ptr()
	try:
		tickets = []
		for offset in (1, 2, 3):
			values = service.features[[1]] + offset
			tickets.append(deferred.submit("features", torch.tensor([1]), values))
			baseline.change_features(torch.tensor([1]), values)
		assert service.catalog.version == 3 and service.counters.tasks == before.tasks
		assert all(not ticket.done() and not ticket.cancel() for ticket in tickets)
		assert service.counters.updates == 0 and storage.store.bytes_used == 0
		batch = deferred.flush()
		assert batch is not None
		results = [ticket.result() for ticket in tickets]
		assert [result.version for result in results] == [3, 3, 3]
		assert [result.input_version for result in results] == [1, 2, 3]
		assert sum(result.tasks for result in results) == service.counters.tasks - before.tasks
		assert service.counters.rows - before.rows < baseline.counters.rows - baseline_before.rows
		assert service.counters.updates == service.counters.input_changes == 3
		assert service.counters.merged_batches == 1 and service.counters.coalesced_affected_rows > 0
		assert service.features.data_ptr() == pointer and not deferred.pending
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		deferred.close()
		plane.close()
		baseline_plane.close()


def test_budget_triggers_flush_before_exposing_older_outputs_and_direct_sync_forces_pending_work():
	service, plane, storage, deferred = _initialized(max_versions=2)
	try:
		first = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		second = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		assert not first.done() and not second.done()
		assert service.catalog.ages(2, torch.tensor([1]), 2).item() == 2
		third = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		assert all(ticket.result().version == 3 for ticket in (first, second, third))
		assert service.catalog.ages(2, torch.tensor([1]), 3).item() == 0
		pending = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		strict = service.change_features(torch.tensor([2]), service.features[[2]] + 0.5)
		assert pending.result().version == 4 and strict.version == 5
		assert service.counters.updates == 5
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		deferred.close()
		plane.close()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_merged_topology_changes_and_strict_queries_use_fresh_dependencies(name):
	service, plane, storage, deferred = _initialized(name)
	try:
		tickets = [
			deferred.submit("insert", torch.tensor([2, 0]), torch.tensor([1, 4])),
			deferred.submit("features", torch.tensor([1]), service.features[[1]] + 2),
			deferred.submit("delete", torch.tensor([1]), torch.tensor([2])),
		]
		with service.query(torch.tensor([2, 3]), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service)[[2, 3]], rtol=2e-5, atol=2e-5)
		assert all(not ticket.done() for ticket in tickets)
		deferred.flush()
		assert service.counters.updates == 3 and service.counters.input_changes == 4
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		deferred.close()
		plane.close()


def test_partial_failure_preserves_applied_batch_for_resume_without_replaying_or_double_credit():
	service, plane, storage, deferred = _initialized()
	apply, calls = service.apply, 0

	def fail(worker, task, inputs):
		nonlocal calls
		calls += 1
		if calls == 2:
			raise RuntimeError("injected partial compute failure")
		return apply(worker, task, inputs)

	try:
		tickets = [deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1) for _ in range(2)]
		applied = service.features.clone()
		service.apply = fail
		with pytest.raises(RuntimeError, match="injected"):
			deferred.flush()
		assert service._pending_update.version == 2 and service.counters.updates == 0
		for ticket in tickets:
			with pytest.raises(RuntimeError, match="injected"):
				ticket.result()
		with pytest.raises(RuntimeError, match="incomplete"):
			deferred.submit("features", torch.tensor([1]), service.features[[1]] + 1)
		service.apply = apply
		assert service.resume_update().version == 2
		assert service.counters.updates == service.counters.input_changes == 2
		assert torch.equal(service.features, applied) and not deferred.pending
		assert service.resume_update().version == 2 and service.counters.updates == 2
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		service.apply = apply
		deferred.close()
		plane.close()


def test_idle_and_shutdown_flush_liveness_and_admission_limits_cover_unpublished_tickets():
	service, plane, storage, deferred = _initialized(idle_seconds=0.01)
	admission = UpdateAdmission(service, max_pending=2)
	try:
		assert (
			admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1).result(10).version == 1
		)
		deferred.idle_seconds = 60
		tickets = [
			admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1) for _ in range(2)
		]
		admission._pool.submit(lambda: None).result(5)
		assert not any(ticket.future.done() for ticket in tickets)
		with pytest.raises(AdmissionFull):
			admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1)
		admission.close()
		assert [ticket.result().version for ticket in tickets] == [3, 3]
		assert admission.counters.completed == 3 and admission.counters.pending == 0
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		admission.close()
		deferred.close()
		plane.close()


def test_sync_admission_flushes_immediately_and_analysis_failure_remains_resumable(monkeypatch):
	service, plane, storage, deferred = _initialized()
	admission = UpdateAdmission(service)
	try:
		pending = admission.submit(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 1)
		strict = admission.sync(UpdateKind.FEATURES, torch.tensor([1]), service.features[[1]] + 2)
		assert strict.version == pending.result().version == 2
		assert strict.input_version == 2 and pending.result().input_version == 1
		analyze = service.affected.analyze

		def fail(**_kwargs):
			raise ValueError("injected analysis failure")

		monkeypatch.setattr(service.affected, "analyze", fail)
		values = service.features[[1]] + 3
		with pytest.raises(ValueError, match="analysis failure"):
			deferred.submit("features", torch.tensor([1]), values)
		assert service.catalog.version == 3 and service._pending_update.version == 3
		torch.testing.assert_close(service.features[[1]], values)
		monkeypatch.setattr(service.affected, "analyze", analyze)
		assert service.resume_update().version == 3 and service.counters.updates == 3
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service))
	finally:
		admission.close()
		deferred.close()
		plane.close()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("budget", [0, 1, 2, 4, 8])
def test_output_age_is_a_shared_version_budget_and_flush_restores_all_layers(name, budget):
	service, plane, storage, deferred = _initialized(name, max_versions=budget)
	nodes = torch.arange(6)
	with service.query(nodes, storage.owner) as initial:
		original = initial.clone()
	base = service.features[[1]].clone()
	tickets = []
	try:
		for version in range(1, budget + 2):
			tickets.append(deferred.submit("features", torch.tensor([1]), base + version))
			if version <= budget:
				assert all(not ticket.done() for ticket in tickets)
				assert service.catalog.ages(2, torch.tensor([3]), version).item() == version
				with service.query(nodes, storage.owner, max_staleness=budget) as stale:
					torch.testing.assert_close(stale, original)
		assert [ticket.result().input_version for ticket in tickets] == list(range(1, budget + 2))
		assert all(ticket.result().version == budget + 1 for ticket in tickets)
		assert not deferred.pending and service.counters.updates == budget + 1
		assert not service.catalog.ages(2, nodes, budget + 1).any()
		with service.query(nodes, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		deferred.close()
		plane.close()


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_overlapping_updates_reinvalidate_a_downstream_row_computed_before_final_flush(name):
	service, plane, storage, deferred = _initialized(name)
	downstream = torch.tensor([3])
	try:
		first = deferred.submit("features", torch.tensor([1]), service.features[[1]] + 3)
		with service.query(downstream, storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service)[downstream], rtol=2e-5, atol=2e-5)
		assert service.catalog.available(2, downstream, 1).all() and not first.done()
		tickets = [
			first,
			deferred.submit("insert", torch.tensor([2]), downstream),
			deferred.submit("features", torch.tensor([2]), service.features[[2]] + 5),
			deferred.submit("delete", torch.tensor([1]), torch.tensor([2])),
		]
		assert not service.catalog.available(2, downstream, 4).any()
		deferred.flush()
		assert all(ticket.result().version == 4 for ticket in tickets)
		assert service.graph.num_edges() == 6
		with service.query(torch.arange(6), storage.owner) as actual:
			torch.testing.assert_close(actual, _current_reference(service), rtol=2e-5, atol=2e-5)
	finally:
		deferred.close()
		plane.close()
