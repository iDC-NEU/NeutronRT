"""Bounded phase summaries and optional CPU traces without tensor snapshots."""

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread, get_ident
from time import perf_counter_ns
from typing import Callable, Dict, Iterator, Optional, TextIO

import psutil


@dataclass
class _Timing:
	count: int = 0
	total: float = 0.0
	minimum: float = math.inf
	maximum: float = 0.0
	failures: int = 0

	def add(self, seconds: float, failed: bool) -> None:
		self.count += 1
		self.total += seconds
		self.minimum = min(self.minimum, seconds)
		self.maximum = max(self.maximum, seconds)
		self.failures += int(failed)

	def snapshot(self) -> dict:
		return {
			"count": self.count,
			"total_seconds": self.total,
			"min_seconds": self.minimum,
			"max_seconds": self.maximum,
			"failures": self.failures,
		}


class Recorder:
	"""Explicit measurement windows exclude startup/warmup and bound series count.

	Phase intervals may overlap, so their totals must not be added to infer wall
	time. Direct duration observations support CUDA-event summaries. Optional trace
	events describe CPU wall intervals only; CUDA timelines use a device profiler.
	"""

	def __init__(
		self,
		*,
		max_series: int = 2048,
		max_trace_events: int = 100_000,
		clock_ns: Callable[[], int] = perf_counter_ns,
	) -> None:
		if max_series < 1 or max_trace_events < 0:
			raise ValueError("Invalid profiler summary or trace limits")
		self.max_series, self.max_trace_events = max_series, max_trace_events
		self._clock = clock_ns
		self._lock = RLock()
		self._active = False
		self._epoch = 0
		self._start = self._end = 0
		self._open = self._dropped = self._trace_events = self._trace_dropped = 0
		self._counters: Dict[str, int] = {}
		self._timings: Dict[str, _Timing] = {}
		self._trace: Optional[TextIO] = None
		self._trace_path: Optional[str] = None

	@property
	def active(self) -> bool:
		return self._active

	def start(self, trace_path: Optional[Path] = None) -> None:
		with self._lock:
			if self._active:
				raise RuntimeError("Profiler measurement window is already active")
			if trace_path is not None:
				trace_path.parent.mkdir(parents=True, exist_ok=True)
				self._trace = trace_path.open("w")
				self._trace.write('{"traceEvents":[\n')
			self._trace_path = str(trace_path) if trace_path is not None else None
			self._epoch += 1
			self._counters.clear()
			self._timings.clear()
			self._open = self._dropped = self._trace_events = self._trace_dropped = 0
			self._start, self._end = self._clock(), 0
			self._active = True

	def _admit_series(self, name: str, *, timing: bool = False) -> bool:
		if name in (self._timings if timing else self._counters):
			return True
		if len(self._counters) + len(self._timings) >= self.max_series:
			self._dropped += 1
			return False
		return True

	def count(self, name: str, value: int = 1) -> None:
		if not self._active:
			return
		with self._lock:
			if self._active and self._admit_series(name):
				self._counters[name] = self._counters.get(name, 0) + value

	def observe(self, name: str, seconds: float, *, failed: bool = False) -> None:
		if not self._active:
			return
		if not math.isfinite(seconds) or seconds < 0:
			raise ValueError("Profile durations must be finite and nonnegative")
		with self._lock:
			if self._active and self._admit_series(name, timing=True):
				self._timings.setdefault(name, _Timing()).add(seconds, failed)

	@contextmanager
	def span(self, name: str) -> Iterator[None]:
		with self._lock:
			if self._active:
				epoch, start = self._epoch, self._clock()
				self._open += 1
			else:
				epoch, start = -1, 0
		failed = False
		try:
			yield
		except BaseException:
			failed = True
			raise
		finally:
			with self._lock:
				if epoch == self._epoch:
					self._open -= 1
					if self._active:
						end = self._clock()
						self.observe(name, (end - start) / 1e9, failed=failed)
						if self._trace is not None:
							self._write_trace(name, start, end, failed)

	def _write_trace(self, name: str, start: int, end: int, failed: bool) -> None:
		if self._trace_events >= self.max_trace_events:
			self._trace_dropped += 1
			return
		assert self._trace is not None
		if self._trace_events:
			self._trace.write(",\n")
		self._trace.write(
			json.dumps(
				{
					"name": name,
					"cat": "cpu.phase",
					"ph": "X",
					"ts": (start - self._start) / 1000,
					"dur": (end - start) / 1000,
					"pid": os.getpid(),
					"tid": get_ident(),
					"args": {"failed": failed},
				}
			)
		)
		self._trace_events += 1

	def snapshot(self) -> dict:
		with self._lock:
			end = self._clock() if self._active else self._end
			return {
				"schema_version": 1,
				"pid": os.getpid(),
				"active": self._active,
				"window_seconds": (end - self._start) / 1e9 if self._epoch else 0.0,
				"counters": dict(self._counters),
				"timings": {name: timing.snapshot() for name, timing in self._timings.items()},
				"open_spans": self._open,
				"series_dropped": self._dropped,
				"trace_path": self._trace_path,
				"trace_events": self._trace_events,
				"trace_events_dropped": self._trace_dropped,
			}

	def stop(self) -> dict:
		with self._lock:
			if not self._active:
				raise RuntimeError("No active profiler measurement window")
			self._end, self._active = self._clock(), False
			if self._trace is not None:
				self._trace.write("\n]}\n")
				self._trace.close()
				self._trace = None
			return self.snapshot()


class MemorySampler:
	"""Sample process RSS with constant memory; GPU allocator peaks are separate."""

	def __init__(self, interval: float = 0.02, *, read: Optional[Callable[[], int]] = None) -> None:
		if interval <= 0:
			raise ValueError("RSS sampling interval must be positive")
		self._read = read or (lambda: psutil.Process().memory_info().rss)
		self.interval = interval
		self._stop = Event()
		self._thread: Optional[Thread] = None
		self.start_bytes = self.peak_bytes = self.end_bytes = self.samples = 0
		self.errors = 0
		self.last_error: Optional[str] = None

	def _sample(self) -> None:
		try:
			value = int(self._read())
			if value < 0:
				raise ValueError("RSS sampler returned a negative byte count")
		except Exception as error:
			self.errors += 1
			self.last_error = f"{type(error).__name__}: {error}"
			return
		self.peak_bytes = max(self.peak_bytes, value)
		self.end_bytes = value
		self.samples += 1

	def start(self) -> None:
		if self._thread is not None:
			raise RuntimeError("RSS sampler already started")
		self._sample()
		self.start_bytes = self.end_bytes
		self._stop.clear()
		self._thread = Thread(target=self._run, name="PROFILE-RSS", daemon=True)
		self._thread.start()

	def _run(self) -> None:
		while not self._stop.wait(self.interval):
			self._sample()

	def stop(self, timeout: Optional[float] = None) -> dict:
		self._stop.set()
		joined = True
		if self._thread is not None:
			join_timeout = max(0.1, min(5.0, self.interval * 4)) if timeout is None else max(0.0, timeout)
			self._thread.join(join_timeout)
			joined = not self._thread.is_alive()
		if joined:
			self._sample()
		return {
			"start_rss_bytes": self.start_bytes,
			"peak_rss_bytes": self.peak_bytes,
			"sampled_peak_rss_bytes": self.peak_bytes,
			"end_rss_bytes": self.end_bytes,
			"start_bytes": self.start_bytes,
			"peak_bytes": self.peak_bytes,
			"end_bytes": self.end_bytes,
			"samples": self.samples,
			"sampling_errors": self.errors,
			"last_sampling_error": self.last_error,
			"interval_seconds": self.interval,
			"sampler_stopped": joined,
		}
