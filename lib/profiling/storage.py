"""Measure independent CPU topology partitions without copying graph data."""

from functools import partial
from typing import Optional

import psutil

from lib.profiling.recorder import MemorySampler


def _rss(actor: psutil.Process) -> int:
	return int(actor.memory_info().rss)


def start(topology) -> tuple[Optional[dict], dict]:
	if topology is None:
		return None, {}
	before = topology.storage.stats()
	samplers: dict[int, MemorySampler] = {}
	if getattr(topology.storage, "remote", False):
		return before, samplers
	for index, process in enumerate(topology.storage.processes):
		actor = psutil.Process(process.pid)
		sampler = MemorySampler(read=partial(_rss, actor))
		sampler.start()
		samplers[index] = sampler
	return before, samplers


def stop(topology, before: Optional[dict], samplers: dict) -> Optional[dict]:
	if before is None:
		return None
	rss = {index: sampler.stop() for index, sampler in samplers.items()}
	after = topology.storage.stats()
	partitions = {}
	for index, value in after.items():
		previous = before[index]
		partitions[index] = {
			"rss": rss.get(
				index,
				{
					"start_bytes": previous.get("rss_bytes", 0),
					"peak_bytes": value.get("rss_bytes", 0),
					"end_bytes": value.get("rss_bytes", 0),
					"samples": 1,
				},
			),
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
		"initialization": topology.initialization,
		"note": (
			"Per-process RSS counts shared pages repeatedly; PSS/USS are end-of-window samples, not peaks. "
			"Shared response bytes are logical payloads, not extra full-graph copies."
		),
	}
