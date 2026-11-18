from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.deferred import DeferredUpdates
from lib.graph.recompute import PlanTooLarge
from lib.profiling.communication import ProfiledCommManager
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from tests.graph.test_serving import _current_reference, _setup


def _runtime(monkeypatch):
	service, plane, storage, _ = _setup("sage")
	service.ensure(torch.arange(6))
	comm = ProfiledCommManager(1, 0, group="not-used", device="cpu")
	transport = SimpleNamespace(compute=Mock(), read=Mock())
	runtime = SimpleNamespace(
		switches=RuntimeFeatures(profiling=True),
		comm=comm,
		plane=SimpleNamespace(transport=transport),
		service=service,
		topology=SimpleNamespace(storage=SimpleNamespace(stats=lambda: {}, processes=[]), initialization={}),
		reset_count=0,
		profile_dir=None,
	)
	monkeypatch.setattr("lib.profiling.controller.start_workers", lambda *args, **kwargs: None)
	monkeypatch.setattr("lib.profiling.controller.stop_workers", lambda *args, **kwargs: [])
	return runtime, plane, storage


def test_cpu_coordinator_windows_measure_completed_work_and_restore_original_methods(monkeypatch):
	runtime, plane, storage = _runtime(monkeypatch)
	service = runtime.service
	original_update = service.change_features
	pointer = service.features.data_ptr()
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			result = service.change_features(torch.tensor([1]), service.features[[1]] + 0.5)
			report = profiler.stop()
			assert report["method"] == "RTEC-UER"
			assert report["valid"] and report["traffic_balanced"] and not report["trace_pass"]
			assert report["engine"]["updates"] == 1 and report["engine"]["input_changes"] == 1
			assert report["engine"]["tasks"] == result.tasks
			assert report["engine"]["completion_window_seconds"] > 0
			assert report["engine"]["completed_changes_per_second"] > 0
			assert report["coordinator"]["timings"]["root.update.features"]["count"] == 1
			assert report["coordinator"]["timings"]["root.affected_analysis"]["count"] >= 2
			assert report["coordinator"]["counters"]["pipeline.compute_thread_ns"] > 0
			assert report["coordinator"]["rss"]["peak_rss_bytes"] > 0
			profiler.start()
			with service.query(torch.arange(6), storage.owner) as actual:
				torch.testing.assert_close(actual, _current_reference(service))
			queries = profiler.stop()
			assert queries["engine"]["updates"] == 0
			assert queries["engine"]["completed_changes_per_second"] is None
		assert "change_features" not in vars(service)
		assert service.change_features == original_update and service.features.data_ptr() == pointer
	finally:
		plane.close()


def test_reset_invalidates_performance_comparison_and_disabled_profiling_fails_before_control_io(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			runtime.reset_count += 1
			report = profiler.stop()
			assert not report["valid"] and report["worker_world_reset"]
			assert report["workers"] == []
		runtime.switches = replace(runtime.switches, profiling=False)
		with pytest.raises(ValueError, match="enabled"):
			RuntimeProfiler(runtime)
	finally:
		plane.close()


def test_failed_execution_invalidates_the_measurement_even_without_a_worker_reset(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			with pytest.raises(ValueError):
				runtime.service.change_features(torch.tensor([-1]), torch.zeros(1, 4))
			report = profiler.stop()
			assert not report["valid"]
			assert report["coordinator"]["timings"]["root.update.features"]["failures"] == 1
	finally:
		plane.close()


@pytest.mark.parametrize("edge_budget,successful", [(7, True), (0, False)])
def test_sizing_retries_preserve_valid_windows_but_irreducible_requests_fail(monkeypatch, edge_budget, successful):
	runtime, plane, _ = _runtime(monkeypatch)
	service = runtime.service
	nodes = torch.arange(6)
	service.catalog.begin_version(1, {1: nodes, 2: nodes})
	service.planner.max_edges = edge_budget
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			if successful:
				service.ensure(nodes)
			else:
				with pytest.raises(PlanTooLarge, match="edge budget"):
					service.ensure(nodes)
			report = profiler.stop()
		assert bool(service.catalog.available(2, nodes, 1).all()) == successful
		assert report["valid"] == successful
		failures = report["coordinator"]["timings"]["root.dependency_plan"]["failures"]
		assert (failures == 0) == successful
	finally:
		plane.close()


def test_failed_query_execution_invalidates_the_window_after_successful_planning(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	service = runtime.service
	nodes = torch.arange(6)
	service.catalog.begin_version(1, {1: nodes, 2: nodes})
	service.apply = Mock(side_effect=RuntimeError("Injected model failure"))
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			with pytest.raises(RuntimeError, match="model failure"):
				service.ensure(nodes)
			report = profiler.stop()
		assert not report["valid"]
		assert report["coordinator"]["timings"]["root.dependency_plan"]["failures"] == 0
		assert report["coordinator"]["timings"]["root.recomputation"]["failures"] == 1
	finally:
		plane.close()


def test_idle_publication_is_fenced_before_the_caller_stops_its_measurement(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	deferred = DeferredUpdates(runtime.service, max_versions=2, idle_seconds=0.01)
	admission = UpdateAdmission(runtime.service)
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			admission.submit(UpdateKind.FEATURES, torch.tensor([1]), runtime.service.features[[1]] + 1).result(10)
			report = profiler.stop()
			assert report["valid"] and report["coordinator"]["open_spans"] == 0
			assert report["publication"]["service_drained"]
			assert report["engine"]["updates"] == report["engine"]["merged_batches"] == 1
			assert report["engine"]["completion_window_seconds"] >= 0.01
			assert report["engine"]["staging_seconds"] > 0
			assert report["coordinator"]["timings"]["root.publish_merged_update"]["count"] == 1
	finally:
		admission.close()
		deferred.close()
		plane.close()


def test_undrained_window_is_invalid_and_stop_does_not_execute_unmeasured_work(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	deferred = DeferredUpdates(runtime.service, max_versions=8, idle_seconds=60)
	try:
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			future = deferred.submit("features", torch.tensor([1]), runtime.service.features[[1]] + 1)
			report = profiler.stop()
			assert not report["valid"] and not future.done()
			assert report["publication"] == {
				"pending_deferred_updates": 1,
				"update_incomplete": False,
				"service_drained": False,
			}
			assert report["engine"]["updates"] == 0
			with pytest.raises(RuntimeError, match="pending update publication"):
				profiler.start()
			deferred.flush()
			assert future.result().version == 1
			profiler.start()
			deferred.submit("features", torch.tensor([1]), runtime.service.features[[1]] + 1)
			deferred.flush()
			report = profiler.stop()
			assert report["valid"] and report["publication"]["service_drained"]
			assert report["engine"]["updates"] == 1
	finally:
		deferred.close()
		plane.close()


def test_attached_admission_window_reports_safe_counter_deltas(monkeypatch):
	runtime, plane, _ = _runtime(monkeypatch)
	admission = UpdateAdmission(runtime.service)
	try:
		with RuntimeProfiler(runtime, admission=admission) as profiler:
			profiler.start()
			admission.sync(UpdateKind.FEATURES, torch.tensor([1]), runtime.service.features[[1]] + 1)
			report = profiler.stop()
		assert report["valid"]
		assert report["admission_window"] == {"admitted": 1, "completed": 1, "failed": 0, "pending_at_stop": 0}
		assert report["instrumentation"]["counter_resets"] == []
		assert "reset_semantics" in report["instrumentation"]
		assert report["policy_window"]["after"]["dispatch"]["mode"] == "round_robin"
		assert report["policy_window"]["counter_resets"] == []
	finally:
		admission.close()
		plane.close()
