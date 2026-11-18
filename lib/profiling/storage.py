"""Measure independent CPU topology partitions without copying graph data."""

from functools import partial
from threading import Event, Thread
from typing import Optional

import psutil

from lib.profiling.recorder import MemorySampler


def _rss(actor: psutil.Process) -> int:
	return int(actor.memory_info().rss)


class RemoteRSSSampler:
	"""Best-effort, low-rate RSS sampling through the storage profile endpoint."""

	def __init__(self, storage, interval: float) -> None:
		if interval <= 0:
			raise ValueError("RSS sampling interval must be positive")
		self.storage, self.interval = storage, interval
		self._stop = Event()
		self._thread: Optional[Thread] = None
		self._start: dict[int, int] = {}
		self._peak: dict[int, int] = {}
		self._end: dict[int, int] = {}
		self._samples: dict[int, int] = {}
		self._errors = 0
		self._last_error: Optional[str] = None
		self._stopped = False

	@property
	def stopped(self) -> bool:
		return self._stopped

	def _sample(self) -> None:
		try:
			values = self.storage.sample_rss(timeout=min(0.5, max(0.05, self.interval)))
		except Exception as error:
			self._errors += 1
			self._last_error = f"{type(error).__name__}: {error}"
			return
		for index, value in values.items():
			try:
				rss = int(value["rss_bytes"])
				if rss < 0:
					raise ValueError("RSS sampler returned a negative byte count")
			except Exception as error:
				self._errors += 1
				self._last_error = f"{type(error).__name__}: {error}"
				continue
			index = int(index)
			self._start.setdefault(index, rss)
			self._peak[index] = max(self._peak.get(index, rss), rss)
			self._end[index] = rss
			self._samples[index] = self._samples.get(index, 0) + 1

	def start(self) -> None:
		if self._thread is not None:
			raise RuntimeError("RSS sampler already started")
		self._stop.clear()
		self._thread = Thread(target=self._run, name="PROFILE-REMOTE-RSS", daemon=True)
		self._thread.start()
		# The first sample belongs to the background sampler. This keeps start()
		# from turning a remote endpoint stall into an unbounded control-path wait.

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
		self._stopped = joined
		return {
			index: {
				"start_rss_bytes": self._start.get(index),
				"peak_rss_bytes": self._peak.get(index),
				"sampled_peak_rss_bytes": self._peak.get(index),
				"end_rss_bytes": self._end.get(index),
				"start_bytes": self._start.get(index),
				"peak_bytes": self._peak.get(index),
				"end_bytes": self._end.get(index),
				"samples": self._samples.get(index, 0),
				"sampling_errors": self._errors,
				"last_sampling_error": self._last_error,
				"interval_seconds": self.interval,
				"sampler_stopped": joined,
			}
			for index in sorted(set(self._start) | set(self._peak) | set(self._end) | set(self._samples))
		}


def start(topology, *, interval: float = 0.5) -> tuple[Optional[dict], dict]:
	if topology is None:
		return None, {}
	before = topology.storage.stats()
	samplers: dict[int | str, MemorySampler | RemoteRSSSampler] = {}
	if getattr(topology.storage, "remote", False):
		remote = RemoteRSSSampler(topology.storage, interval)
		try:
			remote.start()
		except BaseException:
			remote.stop(timeout=0)
			raise
		samplers["remote"] = remote
		return before, samplers
	try:
		for index, process in enumerate(topology.storage.processes):
			actor = psutil.Process(process.pid)
			sampler = MemorySampler(read=partial(_rss, actor))
			sampler.start()
			samplers[index] = sampler
	except BaseException:
		for active_sampler in samplers.values():
			active_sampler.stop(timeout=0)
		raise
	return before, samplers


def stop(topology, before: Optional[dict], samplers: dict) -> Optional[dict]:
	if before is None:
		return None
	remote_sampler = samplers.pop("remote", None)
	remote_rss = remote_sampler.stop() if remote_sampler is not None else {}
	rss = {index: sampler.stop() for index, sampler in samplers.items()}
	after = topology.storage.stats()
	partitions = {}
	for index, value in after.items():
		previous = before[index]
		fallback = {
			"start_rss_bytes": previous.get("rss_bytes"),
			"peak_rss_bytes": None,
			"sampled_peak_rss_bytes": None,
			"end_rss_bytes": value.get("rss_bytes"),
			"samples": 0,
			"sampling_errors": 0,
			"last_sampling_error": None,
			"interval_seconds": None,
			"sampler_stopped": remote_sampler is None or remote_sampler.stopped,
		}
		partitions[index] = {
			"rss": remote_rss.get(index, rss.get(index, fallback)),
			"memory_end": {
				name: value[name]
				for name in (
					"rss_bytes",
					"pss_bytes",
					"uss_bytes",
					"overlay_bytes",
					"overlay_metadata_bytes",
					"overlay_rows",
				)
			},
			"cpu_seconds": value["cpu_seconds"] - previous["cpu_seconds"],
			"shared_response_bytes": value["shared_response_bytes"] - previous["shared_response_bytes"],
			"operations": {
				name: count - previous["operations"].get(name, 0) for name, count in value["operations"].items()
			},
			"seconds": {name: count - previous["seconds"].get(name, 0.0) for name, count in value["seconds"].items()},
			"torch_loaded": value["torch_loaded"],
			"cuda_initialized": value["cuda_initialized"],
			"response_buffers": value["response_buffers"],
		}
	return {
		"partitions": partitions,
		"sampling_complete": all(part["rss"]["sampler_stopped"] for part in partitions.values()),
		"rss_scope": "Remote RSS is a sampled process peak; memory_end is an endpoint/logical snapshot.",
		"initialization": topology.initialization,
		"note": (
			"Per-process RSS counts shared pages repeatedly; sampled peaks are not endpoint values. "
			"PSS/USS and logical overlay bytes are end-of-window samples, not peaks. "
			"Shared response bytes are logical payloads, not extra full-graph copies."
		),
	}
