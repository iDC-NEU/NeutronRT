import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from lib.profiling.recorder import MemorySampler, Recorder


def test_nested_phases_measure_overlap_without_conflating_totals_with_wall_time(tmp_path):
	now = [0]
	recorder = Recorder(clock_ns=lambda: now[0])
	trace = tmp_path / "trace.json"
	recorder.start(trace)
	with recorder.span("update"):
		now[0] += 10**9
		with recorder.span("compile"):
			now[0] += 2 * 10**9
		now[0] += 10**9
	recorder.count("edges", 17)
	report = recorder.stop()
	assert report["window_seconds"] == 4
	assert report["timings"]["update"]["total_seconds"] == 4
	assert report["timings"]["compile"]["total_seconds"] == 2
	assert report["counters"] == {"edges": 17} and report["open_spans"] == 0
	events = json.loads(trace.read_text())["traceEvents"]
	assert [event["name"] for event in events] == ["compile", "update"]
	assert events[0]["dur"] == 2_000_000 and events[1]["dur"] == 4_000_000


def test_start_excludes_warmup_and_old_spans_cannot_leak_into_a_new_window():
	recorder = Recorder()
	recorder.count("warmup", 99)
	with recorder.span("warmup"):
		pass
	recorder.start()
	old = recorder.span("old-window")
	old.__enter__()
	assert recorder.stop()["open_spans"] == 1
	recorder.start()
	old.__exit__(None, None, None)
	recorder.count("measured", 3)
	report = recorder.stop()
	assert report["counters"] == {"measured": 3}
	assert not report["timings"] and report["open_spans"] == 0


def test_failures_are_counted_without_retaining_exception_objects():
	recorder = Recorder()
	recorder.start()
	with pytest.raises(ValueError), recorder.span("failed"):
		raise ValueError("operator failed")
	report = recorder.stop()
	assert report["timings"]["failed"]["failures"] == 1
	assert report["open_spans"] == 0
	json.dumps(report, allow_nan=False)


def test_parallel_updates_do_not_lose_counts_and_series_and_trace_memory_are_bounded(tmp_path):
	recorder = Recorder(max_series=2, max_trace_events=2)
	recorder.start(tmp_path / "bounded.json")
	with ThreadPoolExecutor(max_workers=4) as threads:
		list(threads.map(lambda _: recorder.count("bytes", 8), range(1000)))
	for _ in range(4):
		with recorder.span("operator"):
			pass
	for index in range(100):
		recorder.count(f"excess-{index}")
	report = recorder.stop()
	assert report["counters"]["bytes"] == 8000
	assert len(report["counters"]) + len(report["timings"]) == 2
	assert report["series_dropped"] == 100 and report["trace_events_dropped"] == 2
	assert len(json.loads((tmp_path / "bounded.json").read_text())["traceEvents"]) == 2


def test_memory_sampling_keeps_only_start_peak_and_end_and_joins_its_thread():
	sampled = Event()
	values = iter([10, 40, 20])

	def read():
		value = next(values, 20)
		if value == 40:
			sampled.set()
		return value

	sampler = MemorySampler(interval=0.001, read=read)
	sampler.start()
	assert sampled.wait(2)
	report = sampler.stop()
	assert report["start_rss_bytes"] == 10 and report["peak_rss_bytes"] == 40
	assert report["end_rss_bytes"] == 20 and report["samples"] >= 3
	assert not sampler._thread.is_alive()


def test_counter_and_timing_with_the_same_name_still_respect_the_series_limit():
	recorder = Recorder(max_series=1)
	recorder.start()
	recorder.count("same-name")
	recorder.observe("same-name", 1.0)
	report = recorder.stop()
	assert report["counters"] == {"same-name": 1} and not report["timings"]
	assert report["series_dropped"] == 1
