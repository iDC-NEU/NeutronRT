"""Matched single-device resident replay for NrtInc and native DGL methods."""

import json
import logging
import platform
import resource
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast

import dgl
import torch
import typer
from torch import Tensor

from benchmarks.streaming import _from_wire, apply_update, decode_update
from benchmarks.streaming import run as run_native
from lib.dataset.inference import load_graph
from lib.graph.resident_runtime import (
	ResidentGraphSAGE,
)
from lib.profiling.accuracy import load_test_labels, query_error
from lib.profiling.experiment import inference_identity, latency_summary, source_fingerprint
from lib.profiling.workload import RecordedWorkload


class Method(StrEnum):
	NRT_INC = "NrtInc"
	FULL = "RTEC-Full"
	NS = "RTEC-NS"
	UER = "RTEC-UER"


app = typer.Typer(help=__doc__, pretty_exceptions_show_locals=False)


def _rss_bytes() -> int:
	value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return int(value * (1024 if platform.system() == "Linux" else 1))


def _memory(device: torch.device) -> dict[str, int | None]:
	return {
		"host_peak_rss_bytes_harness_inclusive": _rss_bytes(),
		"gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
		"gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
	}


def _final_state(runtime, graph, features) -> dict[str, Any]:
	source, destination, edge_ids = runtime.graph.edges(form="all")
	expected_source, expected_destination, expected_edge_ids = graph.edges(form="all")
	return {
		"edges_match": bool(
			torch.equal(source.cpu(), expected_source)
			and torch.equal(destination.cpu(), expected_destination)
			and torch.equal(edge_ids.cpu(), expected_edge_ids)
		),
		"features_match": bool(torch.equal(runtime.features.cpu(), features)),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"scope": "resident mutable graph and feature state against an independent CPU replay",
	}


def _fresh(model, graph, features) -> torch.Tensor:
	with torch.inference_mode():
		values = features
		for layer in range(model.num_layers):
			values = model(layer, graph, values)
		return cast(Tensor, values.cpu())


def _run_nrt(
	graph,
	features,
	recording: RecordedWorkload,
	output: Path,
	*,
	device: torch.device,
	capacity_bytes: int,
	labels: torch.Tensor | None,
	test_mask: torch.Tensor | None,
) -> dict[str, Any]:
	model_started = perf_counter()
	model = recording.spec.build()
	model_loading_seconds = perf_counter() - model_started
	if inference_identity(graph, features, recording.spec, model=model) != recording.input_identity:
		raise ValueError("Resident graph, features or model weights differ from the recorded workload")
	oracle_graph, oracle_features = graph.clone(), features.clone()
	output.mkdir(parents=True, exist_ok=False)
	fingerprint = source_fingerprint(output / "sources.zip")
	if device.type == "cuda":
		torch.cuda.init()
		torch.cuda.reset_peak_memory_stats(device)
	placement_started = perf_counter()
	runtime = ResidentGraphSAGE(
		graph,
		features,
		model,
		device=device,
		capacity_bytes=capacity_bytes,
	)
	runtime._sync()
	device_placement_seconds = perf_counter() - placement_started
	initializing = perf_counter()
	bootstrap = runtime.bootstrap()
	initialization_seconds = perf_counter() - initializing
	initialization_memory = _memory(device)
	oracle_model = recording.spec.build()
	verification_nodes = recording.verification_nodes

	def check(nodes: torch.Tensor) -> dict[str, Any]:
		actual = runtime.query(nodes)
		expected = _fresh(oracle_model, oracle_graph, oracle_features)[nodes]
		torch.testing.assert_close(actual, expected, atol=recording.atol, rtol=recording.rtol)
		return query_error(
			actual,
			expected,
			labels=labels[nodes] if labels is not None else None,
			test_mask=test_mask[nodes] if test_mask is not None else None,
		) | {"exact_equivalence_required": True}

	checks = [check(verification_nodes)]
	warmup = []
	warmup_updates = [decode_update(request) for request in recording.warmup]
	steps = [(decode_update(request), _from_wire(query, 64 * 1024**2)) for request, query in recording.steps]
	for decoded in warmup_updates:
		warmup.append(runtime.update(decoded))
		apply_update(oracle_graph, oracle_features, decoded)
		checks.append(check(verification_nodes))

	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
	updates: list[dict[str, Any]] = []
	queries: list[float] = []
	measured_seconds = 0.0
	harness_started = perf_counter()
	for decoded, query_nodes in steps:
		step_started = perf_counter()
		update = runtime.update(decoded)
		query_started = perf_counter()
		if query_nodes.numel():
			runtime.query(query_nodes)
			queries.append(perf_counter() - query_started)
		else:
			queries.append(0.0)
		service_seconds = perf_counter() - step_started
		phases = dict(update["phase_seconds"])
		if abs(sum(phases.values()) - update["response_seconds"]) > 1e-6:
			raise RuntimeError("Resident update phase timings do not partition the update response")
		update = {
			**update,
			"phase_seconds": phases,
			"response_seconds": update["response_seconds"],
			"query_seconds": queries[-1],
			"service_seconds": service_seconds,
		}
		updates.append(update)
	measured_seconds = sum(item["service_seconds"] for item in updates)
	memory = _memory(device)
	oracle_replay_started = perf_counter()
	for decoded, _ in steps:
		apply_update(oracle_graph, oracle_features, decoded)
	oracle_replay_seconds = perf_counter() - oracle_replay_started
	harness_elapsed_seconds = perf_counter() - harness_started

	final_actual = runtime.query(torch.arange(graph.num_nodes(), dtype=torch.int64))
	final_expected = _fresh(oracle_model, oracle_graph, oracle_features)
	torch.testing.assert_close(final_actual, final_expected, atol=recording.atol, rtol=recording.rtol)
	quality = query_error(final_actual, final_expected, labels=labels, test_mask=test_mask)
	state = _final_state(runtime, oracle_graph, oracle_features)
	if not state["edges_match"] or not state["features_match"]:
		raise RuntimeError("Resident replay final graph or feature state differs from the CPU oracle")
	return {
		"driver_schema": 1,
		"implementation": "NeutronRT-NrtInc-Resident",
		"method": Method.NRT_INC.value,
		"comparison_group": "resident_runtime",
		"backend_role": "reusable_single_device_incremental_runtime",
		"actual_pipeline": False,
		"execution_mode": "single_device_resident_sparse_aggregate",
		"scope": (
			f"{device.type.upper()}-resident graph, features, weights, embeddings and projected aggregate state; "
			"CPU control and query return only"
		),
		"remote_dispatch": False,
		"remote_routing": False,
		"cpu_feature_fetch": False,
		"source_fingerprint": fingerprint,
		"producer_source_fingerprint": recording.producer_source,
		"input_identity": recording.input_identity,
		"trace_digest": recording.digest,
		"validation": {"initial_input_identity_verified": True, "trace_digest_preserved": True},
		"repetition": recording.repetition,
		"spec": asdict(replace(recording.spec, checkpoint=None)),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"device": str(device),
		"batch_size": None,
		"residency": {
			"device": str(device),
			"capacity_bytes": capacity_bytes,
			"admission_estimated_bytes": runtime.admission.estimated_bytes,
			"resident_state_bytes": runtime.admission.resident_bytes,
			"logical_bound_bytes": runtime.capacity_policy.reserved_bytes,
			"reserved_edge_count": runtime.capacity_policy.reserved_edge_count,
			"physical_admission_checks": runtime.capacity_policy.physical_checks,
			"gpu_peak_allocated_bytes": memory["gpu_peak_allocated_bytes"],
			"gpu_peak_reserved_bytes": memory["gpu_peak_reserved_bytes"],
			"physical_peak_bytes": memory["gpu_peak_reserved_bytes"],
			"admission_semantics": (
				"shared logical envelope with one initial physical check and growth checks only outside the envelope"
			),
			"physical_hard_cap_enforced": False,
		},
		"initialization": {"seconds": initialization_seconds, "bootstrap": bootstrap, "memory": initialization_memory},
		"loading": {
			"model_seconds": model_loading_seconds,
			"device_placement_seconds": device_placement_seconds,
			"scope": "recorded model construction and checkpoint load; excluded from measured updates",
		},
		"warmup": warmup,
		"correctness": checks,
		"updates": updates,
		"runtime_counters": dict(runtime.counters),
		"aggregate_dtype": str(runtime.aggregate_dtype),
		"query_count": len(queries),
		"query_rows": sum(query.numel() for _, query in steps),
		"query_latency": latency_summary(queries),
		"window_seconds": measured_seconds,
		"response_seconds": sum(item["response_seconds"] for item in updates) / len(updates),
		"completed_changes_per_second": sum(item["changes"] for item in updates) / measured_seconds,
		"harness_elapsed_seconds": harness_elapsed_seconds,
		"oracle_replay_seconds": oracle_replay_seconds,
		"oracle_replay_scope": "CPU oracle mutations replayed after the last measured service call",
		"final_quality": quality,
		"final_oracle": {
			"rows": graph.num_nodes(),
			"official_test_rows": quality.get("labelled_test_rows"),
			"exact_embedding_equivalence": True,
			"matching_argmax": quality["prediction_disagreement"] == 0.0,
			"accuracy_loss_percentage_points": quality.get("accuracy_loss_percentage_points"),
		},
		"final_state_validation": state,
		"timing_semantics": {
			"exclusive_additivity": True,
			"phase_keys": [
				"input_request_transfer",
				"admission",
				"method_preparation",
				"mutation_index",
				"affected_delta_planning",
				"execution",
				"publication_output",
				"residual",
			],
			"cuda_fences": "one fence at each dependent stage boundary and no per-row/scalar barrier",
			"measured_window": "sum of update-plus-recorded-query service intervals only",
			"response_scope": "resident update-to-publication interval; recorded query is separate",
			"oracle_scope": (
				"all-node fresh CPU oracle, verification and oracle mutation are outside measured intervals"
			),
			"decode_scope": "all recorded updates and queries decoded before the measured window",
		},
		"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "dgl": dgl.__version__},
		"valid": True,
	}


@app.command()
def main(
	data: Annotated[Path, typer.Option(exists=True)],
	workload: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
	output: Annotated[Path, typer.Option(help="New directory for the replay report")],
	checkpoint: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	labels: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	test_split: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
	method: Annotated[Method, typer.Option()] = Method.NRT_INC,
	fanout: Annotated[int, typer.Option(min=1)] = 5,
	device: Annotated[str, typer.Option()] = "cuda:0",
	capacity_bytes: Annotated[int, typer.Option(min=1, help="Hard resident admission capacity")] = 8 * 1024**3,
	batch_size: Annotated[int, typer.Option(min=1)] = 8192,
	threads: Annotated[int, typer.Option(min=1)] = 4,
) -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	torch.set_num_threads(threads)
	loading_started = perf_counter()
	recording = RecordedWorkload.load(workload, checkpoint=checkpoint)
	if recording.spec.name != "sage":
		raise typer.BadParameter("Resident replay currently supports mean GraphSAGE")
	device_ = torch.device(device)
	if device_.type not in ("cpu", "cuda") or (device_.type == "cuda" and device_.index is None):
		raise typer.BadParameter("Resident replay requires cpu or an explicit CUDA device, such as cuda:0")
	graph = load_graph(data)
	features = graph.ndata.pop("feat")
	labels_value, test_mask = load_test_labels(labels, test_split, graph.num_nodes())
	input_loading_seconds = perf_counter() - loading_started
	if method is Method.NRT_INC:
		report = _run_nrt(
			graph,
			features,
			recording,
			output,
			device=device_,
			capacity_bytes=capacity_bytes,
			labels=labels_value,
			test_mask=test_mask,
		)
	else:
		model = recording.spec.build()
		if inference_identity(graph, features, recording.spec, model=model) != recording.input_identity:
			raise ValueError("Resident graph, features or model weights differ from the recorded workload")
		report = run_native(
			graph,
			features,
			recording,
			output,
			fanout=fanout if method is Method.NS else None,
			device=device_,
			output_device=device_,
			batch_size=batch_size,
			output_budget_bytes=capacity_bytes,
			method=method.value,
			labels=labels_value,
			test_mask=test_mask,
			resident=True,
			resident_capacity_bytes=capacity_bytes,
		)
	report["loading"] = {
		**report.get("loading", {}),
		"graph_features_labels_workload_seconds": input_loading_seconds,
		"scope": "workload/graph/features/labels ingress plus model load; excluded from measured updates",
	}
	(output / "trial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
	app()
