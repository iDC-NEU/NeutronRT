"""Replay the framework's immutable workload on conventional DGL references."""

import json
import logging
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import dgl
import torch

from benchmarks.dgl import StreamingRecompute
from lib.profiling.accuracy import query_error, targeted_sample
from lib.profiling.experiment import TraceUpdate, inference_identity, latency_summary, source_fingerprint
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _from_wire


def decode_update(request: wire.Update) -> TraceUpdate:
	"""Decode recorded GraphSAGE inputs outside the standalone timing window."""
	if request.HasField("attributes"):
		raise ValueError("The mean GraphSAGE reference does not consume edge attributes")
	kinds = {wire.Update.FEATURES: "features", wire.Update.INSERT: "insert", wire.Update.DELETE: "delete"}
	if request.kind not in kinds:
		raise ValueError("Unsupported recorded update kind")
	return TraceUpdate(
		kinds[request.kind], _from_wire(request.first, 64 * 1024**2), _from_wire(request.second, 64 * 1024**2)
	)


def run(
	graph,
	features,
	recording: RecordedWorkload,
	output: Path,
	*,
	fanout: int | None,
	device: torch.device,
	output_device: torch.device,
	batch_size: int,
) -> dict[str, Any]:
	model = recording.spec.build()
	if inference_identity(graph, features, recording.spec, model=model) != recording.input_identity:
		raise ValueError("Baseline graph, features or model weights differ from the recorded workload")
	if recording.initialization not in ("all", "queries"):
		raise ValueError("Unsupported recorded initialization mode")
	if not recording.steps or not recording.verification_nodes.numel():
		raise ValueError("A streaming comparison requires measured steps and verification rows")
	output.mkdir(parents=True, exist_ok=False)
	fingerprint = source_fingerprint(output / "sources.zip")
	model.to(device)
	runner = StreamingRecompute(
		graph, features, model, fanout=fanout, batch_size=batch_size, output_device=output_device
	)
	oracle = recording.spec.build()
	verification_nodes = recording.verification_nodes
	decoding = perf_counter()
	warmup_updates = [decode_update(request) for request in recording.warmup]
	steps = [(decode_update(request), _from_wire(query, 64 * 1024**2)) for request, query in recording.steps]
	decode_seconds = perf_counter() - decoding
	torch.manual_seed(recording.seed)
	dgl.seed(recording.seed)

	def check():
		started = perf_counter()
		actual = runner.query(verification_nodes)
		with torch.inference_mode():
			expected = features
			for layer in range(oracle.num_layers):
				expected = oracle(layer, graph, expected)
			expected = expected[verification_nodes]
		if fanout is None:
			torch.testing.assert_close(actual, expected, atol=recording.atol, rtol=recording.rtol)
		result = query_error(actual, expected)
		result["exact_equivalence_required"] = fanout is None
		result["oracle_seconds"] = perf_counter() - started
		return result

	initializing = perf_counter()
	if recording.initialization == "all":
		runner.compute(torch.arange(graph.num_nodes()), progress=True)
	runner.synchronize()
	initialization_seconds = perf_counter() - initializing
	checks = [check()]
	warmup = []
	for request in warmup_updates:
		warmup.append(runner.update(request))
		checks.append(check())
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
	updates, queries = [], []
	started = perf_counter()
	for index, (request, nodes) in enumerate(steps):
		updates.append(runner.update(request))
		if nodes.numel():
			query_started = perf_counter()
			runner.query(nodes)
			queries.append(perf_counter() - query_started)
		logging.info("DGL update %d: response_seconds=%.6f", index + 1, updates[-1]["response_seconds"])
	runner.synchronize()
	elapsed = perf_counter() - started
	peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
	changed = []
	for request, _ in recording.steps:
		changed.append(_from_wire(request.first, 64 * 1024**2))
		if request.kind != wire.Update.FEATURES:
			changed.append(_from_wire(request.second, 64 * 1024**2))
	if changed:
		verification_nodes = targeted_sample(
			graph, verification_nodes, torch.cat(changed), depth=recording.spec.layers, seed=recording.seed
		)
	checks.append(check())
	report = {
		"driver_schema": 1,
		"implementation": "DGL-RTEC-Full" if fanout is None else "DGL-RTEC-NS",
		"scope": "standalone recomputation including topology, analysis, compute, output storage and CPU queries",
		"source_fingerprint": fingerprint,
		"producer_source_fingerprint": recording.producer_source,
		"input_identity": recording.input_identity,
		"trace_digest": recording.digest,
		"repetition": recording.repetition,
		"spec": asdict(replace(recording.spec, checkpoint=None)),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"fanout": fanout,
		"sampling_seed": recording.seed,
		"batch_size": batch_size,
		"device": str(device),
		"output_device": str(output_device),
		"output_bytes": runner.outputs.numel() * runner.outputs.element_size(),
		"input_decode_seconds": decode_seconds,
		"initialization": {"mode": recording.initialization, "seconds": initialization_seconds},
		"warmup": warmup,
		"correctness": checks,
		"updates": updates,
		"query_latency": latency_summary(queries),
		"window_seconds": elapsed,
		"completed_changes_per_second": sum(item["changes"] for item in updates) / elapsed,
		"response_seconds": sum(item["response_seconds"] for item in updates) / len(updates),
		"environment": {
			"python": platform.python_version(),
			"torch": str(torch.__version__),
			"dgl": dgl.__version__,
			"threads": torch.get_num_threads(),
			"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
			"peak_gpu_allocated_bytes": peak_gpu_bytes,
		},
		"valid": True,
	}
	(output / "trial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
	return report
