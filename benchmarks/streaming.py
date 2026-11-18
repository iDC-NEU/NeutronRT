"""Replay the framework's immutable workload on conventional DGL references."""

import hashlib
import importlib
import json
import logging
import platform
import resource
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import dgl
import torch

from benchmarks.dgl import (
	OFFICIAL_DGL_COMMIT,
	OFFICIAL_DGL_RELEASE,
	UER_SOURCE_COMMIT,
	UER_SOURCE_PATH,
	UER_SOURCE_SHA256,
	StreamingRecompute,
	UERRecompute,
	_resident_logical_bound,
	apply_update,
)
from lib.graph.capacity import CapacityAdmissionError, CapacityPolicy
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


def _module_hash(module) -> str:
	return hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()


def _rss_bytes() -> int:
	value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return int(value * (1024 if platform.system() == "Linux" else 1))


def _memory_snapshot(device: torch.device) -> dict[str, int | None]:
	return {
		"host_peak_rss_bytes_harness_inclusive": _rss_bytes(),
		"gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
		"gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
	}


def _logical_clone_bytes(graph, features) -> dict[str, int]:
	id_bytes = torch.empty((), dtype=graph.idtype).element_size()
	return {
		"feature_tensor_bytes": features.numel() * features.element_size(),
		"edge_endpoint_tensor_bytes": graph.num_edges() * 2 * id_bytes,
		"edge_id_tensor_bytes": graph.num_edges() * id_bytes,
	}


def _resident_graph_index_bytes(graph) -> int:
	"""Conservative COO/CSR/CSC/EID logical storage bound for reporting."""
	index_bytes = 8
	nodes, edges = int(graph.num_nodes()), int(graph.num_edges())
	return 2 * edges * index_bytes + 2 * (nodes + 1 + edges) * index_bytes + edges * index_bytes


def _sampler_module_hashes() -> dict[str, str]:
	modules = [dgl, dgl.dataloading, importlib.import_module("dgl.dataloading.neighbor_sampler")]
	return {module.__name__: _module_hash(module) for module in modules}


def _phase(phase_file: Path | None, name: str, **fields: Any) -> None:
	if phase_file is None:
		return
	phase_file.parent.mkdir(parents=True, exist_ok=True)
	temporary = phase_file.with_name(f".{phase_file.name}.tmp")
	temporary.write_text(json.dumps({"phase": name, **fields}, allow_nan=False) + "\n")
	temporary.replace(phase_file)


def _validate_final_state(graph, features, oracle_graph, oracle_features) -> dict[str, Any]:
	source, destination, edge_ids = graph.edges(form="all")
	expected_source, expected_destination, expected_edge_ids = oracle_graph.edges(form="all")
	return {
		"edges_match": bool(
			torch.equal(source.cpu(), expected_source.cpu())
			and torch.equal(destination.cpu(), expected_destination.cpu())
			and torch.equal(edge_ids.cpu(), expected_edge_ids.cpu())
		),
		"features_match": bool(torch.equal(features.cpu(), oracle_features.cpu())),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"scope": "final mutable graph and feature state against an independent DGL clone",
	}


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
	output_budget_bytes: int | None = None,
	method: str | None = None,
	phase_file: Path | None = None,
	labels: torch.Tensor | None = None,
	test_mask: torch.Tensor | None = None,
	resident: bool = False,
	resident_capacity_bytes: int | None = None,
) -> dict[str, Any]:
	method = method or ("RTEC-Full" if fanout is None else "RTEC-NS")
	normalized_method = method.replace("_", "-").lower()
	if normalized_method not in {"rtec-full", "rtec-ns", "rtec-uer"}:
		raise ValueError(f"Unsupported DGL baseline method {method!r}")
	if normalized_method == "rtec-full" and fanout is not None:
		raise ValueError("RTEC-Full requires fanout=None")
	if normalized_method == "rtec-ns" and fanout is None:
		raise ValueError("RTEC-NS requires a positive fanout")
	if normalized_method == "rtec-uer" and fanout is not None:
		raise ValueError("RTEC-UER is the exact full-neighbor reuse baseline and requires fanout=None")
	model_started = perf_counter()
	model = recording.spec.build()
	model_loading_seconds = perf_counter() - model_started
	if inference_identity(graph, features, recording.spec, model=model) != recording.input_identity:
		raise ValueError("Baseline graph, features or model weights differ from the recorded workload")
	if recording.initialization not in ("all", "queries"):
		raise ValueError("Unsupported recorded initialization mode")
	if not recording.steps or not recording.verification_nodes.numel():
		raise ValueError("A streaming comparison requires measured steps and verification rows")
	if resident and (device.type != "cuda" or output_device != device):
		raise ValueError("Resident native replay requires an explicit CUDA device and CUDA output cache")
	if resident and (resident_capacity_bytes is None or resident_capacity_bytes < 1):
		raise ValueError("Resident native replay requires an explicit positive capacity")
	output.mkdir(parents=True, exist_ok=False)
	phase_fields = {
		"method": {"rtec-full": "RTEC-Full", "rtec-ns": "RTEC-NS", "rtec-uer": "RTEC-UER"}[normalized_method],
		"repetition": recording.repetition,
		"trace_digest": recording.digest,
		"backend_identity": f"dgl-{OFFICIAL_DGL_RELEASE}-{OFFICIAL_DGL_COMMIT}",
	}
	_phase(phase_file, "setup", **phase_fields)
	_phase(phase_file, "loading", **phase_fields)
	fingerprint = source_fingerprint(output / "sources.zip")
	oracle_graph = graph.clone()
	oracle_features = features.clone()
	oracle_clone_bytes = _logical_clone_bytes(oracle_graph, oracle_features)
	torch.manual_seed(recording.seed)
	dgl.seed(recording.seed)
	placement_started = perf_counter()
	model.to(device)
	if resident:
		graph = graph.to(device)
		features = features.to(device)
	capacity_policy = None
	if resident:
		initial_bound = _resident_logical_bound(graph, features, model)
		try:
			capacity_policy = CapacityPolicy.create(
				int(resident_capacity_bytes or 0),
				device,
				initial_bound,
				graph.num_edges(),
				lambda edges: _resident_logical_bound(graph, features, model, edge_count=edges),
			)
		except CapacityAdmissionError as exc:
			raise ValueError(str(exc)) from exc
	runner_type = UERRecompute if normalized_method == "rtec-uer" else StreamingRecompute
	runner = runner_type(
		graph,
		features,
		model,
		fanout=fanout,
		batch_size=batch_size,
		output_device=output_device,
		sampling_seed=recording.seed,
		output_budget_bytes=output_budget_bytes,
		capacity_policy=capacity_policy,
		capacity_estimator=(lambda edges: _resident_logical_bound(graph, features, model, edge_count=edges))
		if resident
		else None,
	)
	runner.synchronize()
	device_placement_seconds = perf_counter() - placement_started
	oracle = recording.spec.build()
	verification_nodes = recording.verification_nodes
	decoding = perf_counter()
	warmup_updates = [decode_update(request) for request in recording.warmup]
	steps = [(decode_update(request), _from_wire(query, 64 * 1024**2)) for request, query in recording.steps]
	decode_seconds = perf_counter() - decoding

	def check(actual=None):
		started = perf_counter()
		if actual is None:
			actual = runner.query(verification_nodes)
		with torch.inference_mode():
			expected = oracle_features
			for layer in range(oracle.num_layers):
				expected = oracle(layer, oracle_graph, expected)
			expected = expected[verification_nodes]
		exact = normalized_method in {"rtec-full", "rtec-uer"}
		if exact:
			torch.testing.assert_close(actual, expected, atol=recording.atol, rtol=recording.rtol)
		result = query_error(
			actual,
			expected,
			labels=labels[verification_nodes] if labels is not None else None,
			test_mask=test_mask[verification_nodes] if test_mask is not None else None,
		)
		result["exact_equivalence_required"] = exact
		result["oracle_seconds"] = perf_counter() - started
		return result

	_phase(phase_file, "initialization", **phase_fields)
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
	initialization_timing: dict[str, Any] | None
	initializing = perf_counter()
	if recording.initialization == "all":
		initialization_timing = runner.initialize(progress=True)
		runner.synchronize()
		initialization_seconds = perf_counter() - initializing
		initialization_memory = _memory_snapshot(device)
		checks = [check()]
	else:
		# UER needs every retained intermediate row initialized before an
		# affected-set update. Its first query therefore performs a full cache
		# bootstrap, while Full/NS keep selected-output lazy initialization.
		initialization_timing = {
			"status": (
				"lazy_full_cache_bootstrap_required"
				if normalized_method == "rtec-uer"
				else "lazy_selected_output_initialization"
			)
		}
		first_actual = runner.query(verification_nodes)
		runner.synchronize()
		initialization_seconds = perf_counter() - initializing
		initialization_memory = _memory_snapshot(device)
		checks = [check(first_actual)]
	if recording.initialization == "queries":
		initialization_timing = {"status": "lazy_selected_output_initialization", **runner.last_timing}
		if normalized_method == "rtec-uer":
			initialization_timing["status"] = "lazy_full_cache_bootstrap_required"
	warmup = []
	_phase(phase_file, "warmup", **phase_fields)
	for request in warmup_updates:
		warmup.append(runner.update(request))
		apply_update(oracle_graph, oracle_features, request)
		checks.append(check())
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
	updates, queries = [], []
	measured_seconds = 0.0
	harness_started = perf_counter()
	for index, (request, nodes) in enumerate(steps):
		_phase(phase_file, "update", index=index, **phase_fields)
		step_started = perf_counter()
		update = runner.update(request)
		if nodes.numel():
			query_started = perf_counter()
			runner.query(nodes)
			queries.append(perf_counter() - query_started)
		else:
			queries.append(0.0)
		if resident:
			service_seconds = perf_counter() - step_started
			phases = dict(update["phase_seconds"])
			if abs(sum(phases.values()) - update["response_seconds"]) > 1e-6:
				raise RuntimeError("Native update phase timings do not partition the update response")
			update = {
				**update,
				"phase_seconds": phases,
				"response_seconds": update["response_seconds"],
				"query_seconds": queries[-1],
				"service_seconds": service_seconds,
			}
		updates.append(update)
		measured_seconds += perf_counter() - step_started
		logging.info("DGL update %d: response_seconds=%.6f", index + 1, updates[-1]["response_seconds"])
	runner.synchronize()
	elapsed = measured_seconds
	measured_memory = _memory_snapshot(device)
	oracle_replay_started = perf_counter()
	for request, _ in steps:
		apply_update(oracle_graph, oracle_features, request)
	oracle_replay_seconds = perf_counter() - oracle_replay_started
	harness_elapsed_seconds = perf_counter() - harness_started
	changed = []
	for request, _ in recording.steps:
		changed.append(_from_wire(request.first, 64 * 1024**2))
		if request.kind != wire.Update.FEATURES:
			changed.append(_from_wire(request.second, 64 * 1024**2))
	if changed:
		verification_nodes = targeted_sample(
			oracle_graph, verification_nodes, torch.cat(changed), depth=recording.spec.layers, seed=recording.seed
		)
	checks.append(check())
	final_oracle_started = perf_counter()
	final_nodes = torch.arange(graph.num_nodes(), dtype=torch.int64, device=runner.ready.device)
	final_actual = runner.query(final_nodes)
	with torch.inference_mode():
		final_expected = oracle_features
		for layer in range(oracle.num_layers):
			final_expected = oracle(layer, oracle_graph, final_expected)
		final_expected = final_expected.cpu()
	exact = normalized_method in {"rtec-full", "rtec-uer"}
	if exact:
		torch.testing.assert_close(final_actual, final_expected, atol=recording.atol, rtol=recording.rtol)
	final_quality = query_error(final_actual, final_expected, labels=labels, test_mask=test_mask)
	final_quality["exact_equivalence_required"] = exact
	final_quality["oracle_seconds"] = perf_counter() - final_oracle_started
	final_state = _validate_final_state(graph, features, oracle_graph, oracle_features)
	if not final_state["edges_match"] or not final_state["features_match"]:
		raise RuntimeError("Native DGL replay final graph or feature state differs from the recorded trace")
	report = {
		"driver_schema": 1,
		"implementation": {
			"rtec-full": "DGL-RTEC-Full",
			"rtec-ns": "DGL-RTEC-NS",
			"rtec-uer": "DGL-RTEC-UER",
		}[normalized_method],
		"comparison_group": "framework_ablation" if normalized_method == "rtec-uer" else "dedicated_reference",
		"backend_role": (
			"author_layerwise_uer_adaptation" if normalized_method == "rtec-uer" else "official_dgl_reference"
		),
		"actual_pipeline": False,
		"resident_execution": resident,
		"execution_mode": "sequential_synchronized_stage_boundaries",
		"scope": "standalone recomputation including topology, analysis, compute, output storage and CPU queries",
		"comparison_scope": "direct local-memory DGL reference; not mapped/remote runtime execution",
		"timing_semantics": {
			"stage_scope": ("wall-clock intervals around native sampler, transfer, compute and output-copy stages"),
			"cuda_synchronization": "synchronized at transfer and compute boundaries",
			"exclusive_additivity": resident,
			"phase_keys": [
				"input_request_transfer",
				"admission",
				"mutation_index",
				"method_preparation",
				"affected_delta_planning",
				"execution",
				"publication_output",
				"residual",
			],
			"resident_phase_scope": (
				"native GPU replay exposes exclusive response phases from shared request "
				"preparation through publication"
				if resident
				else "not_applicable"
			),
			"measured_window": (
				"sum of per-step update-plus-recorded-query service intervals; oracle mutation/replay "
				"is outside each interval"
			),
			"response_scope": "update-to-publication interval; recorded query service is reported separately",
			"decode_scope": "all recorded updates and queries decoded before the measured window",
			"oracle_mutation_scope": (
				"independent clone mutation and correctness checks are excluded from measured window"
			),
			"publication": (
				"standalone final output index_copy publication"
				if normalized_method == "rtec-uer"
				else "not_applicable_standalone"
			),
		},
		"source_fingerprint": fingerprint,
		"producer_source_fingerprint": recording.producer_source,
		"input_identity": recording.input_identity,
		"trace_digest": recording.digest,
		"validation": {
			"initial_input_identity_verified": True,
			"trace_digest_preserved": True,
			"model_weights_scope": "included in input_identity",
		},
		"repetition": recording.repetition,
		"spec": asdict(replace(recording.spec, checkpoint=None)),
		"nodes": graph.num_nodes(),
		"edges": graph.num_edges(),
		"fanout": fanout,
		"sampling_seed": recording.seed,
		"sampling": {
			"sampler": type(runner.sampler).__name__,
			"fanout": fanout,
			"seed": recording.seed,
			"seed_scope": "one global DGL seed set before replay; no per-operation reseeding",
			"batch_size": batch_size,
			"topology_version": runner.sampler_generation,
			"policy_identity": (
				"author_layerwise_uer_adaptation; native one-layer sampler per affected layer"
				if normalized_method == "rtec-uer"
				else "native_dgl_v2.4.0_sampler; batch-size-dependent sampled edges"
			),
		},
		"batch_size": batch_size,
		"device": str(device),
		"output_device": str(output_device),
		"residency": {
			"graph_device": str(graph.device),
			"feature_device": str(features.device),
			"model_device": str(next(model.parameters()).device),
			"capacity_bytes": resident_capacity_bytes,
			"resident_state_bytes": (
				features.numel() * features.element_size()
				+ runner.outputs.numel() * runner.outputs.element_size()
				+ sum(value.numel() * value.element_size() for value in model.parameters())
				+ _resident_graph_index_bytes(graph)
				+ runner.ready.numel() * runner.ready.element_size()
				+ (
					sum(value.numel() * value.element_size() for value in runner.states[1:])
					if isinstance(runner, UERRecompute) and runner.states is not None
					else 0
				)
			),
			"remote_dispatch": False,
			"remote_routing": False,
			"cpu_feature_fetch": not resident,
			"gpu_peak_allocated_bytes": measured_memory["gpu_peak_allocated_bytes"],
			"gpu_peak_reserved_bytes": measured_memory["gpu_peak_reserved_bytes"],
			"logical_bound_bytes": _resident_logical_bound(graph, features, model) if resident else None,
			"physical_peak_bytes": measured_memory["gpu_peak_reserved_bytes"],
			"admission_semantics": (
				"shared logical envelope with one initial physical check and growth checks only outside the envelope"
			),
			"physical_admission_checks": capacity_policy.physical_checks if capacity_policy is not None else 0,
			"physical_hard_cap_enforced": False,
		},
		"output_bytes": runner.outputs.numel() * runner.outputs.element_size(),
		"input_decode_seconds": decode_seconds,
		"loading": {
			"model_seconds": model_loading_seconds,
			"device_placement_seconds": device_placement_seconds,
			"scope": "recorded model construction and checkpoint load; excluded from measured updates",
		},
		"initialization": {
			"mode": recording.initialization,
			"seconds": initialization_seconds,
			"timing": initialization_timing,
			"memory": initialization_memory,
			"scope": (
				"official DGL layerwise bootstrap for all mode; "
				+ (
					"full retained-state bootstrap on first query for UER"
					if normalized_method == "rtec-uer"
					else "selected-output lazy compute for queries mode"
				)
			),
		},
		"warmup": warmup,
		"correctness": checks,
		"updates": updates,
		"query_count": len(queries),
		"query_rows": sum(nodes.numel() for _, nodes in steps),
		"query_latency": latency_summary(queries),
		"final_quality": final_quality,
		"final_oracle": {
			"rows": graph.num_nodes(),
			"official_test_rows": final_quality.get("labelled_test_rows"),
			"exact_embedding_equivalence": exact,
			"matching_argmax": final_quality["prediction_disagreement"] == 0.0,
			"accuracy_loss_percentage_points": final_quality.get("accuracy_loss_percentage_points"),
		},
		"window_seconds": elapsed,
		"completed_changes_per_second": sum(item["changes"] for item in updates) / elapsed,
		"response_seconds": sum(item["response_seconds"] for item in updates) / len(updates),
		"harness_elapsed_seconds": harness_elapsed_seconds,
		"oracle_replay_seconds": oracle_replay_seconds,
		"oracle_replay_scope": "CPU oracle mutations replayed after the last measured service call",
		"final_state_validation": final_state,
		"backend": {
			"upstream_release": OFFICIAL_DGL_RELEASE,
			"upstream_commit": OFFICIAL_DGL_COMMIT,
			"installed_dgl_version": dgl.__version__,
			"installed_module_hashes": {
				"dgl_module_file_sha256": _module_hash(dgl),
				"torch_module_file_sha256": _module_hash(torch),
			},
			"sampler_module_hashes": _sampler_module_hashes(),
			"sampler_backend": type(runner.sampler).__name__,
			"uer_provenance": (
				{
					"identity": "author_layerwise_uer_adaptation_of_lambdaGrapher_inspired_Cache",
					"source_commit": UER_SOURCE_COMMIT,
					"source_path": UER_SOURCE_PATH,
					"source_sha256": UER_SOURCE_SHA256,
					"official_dgl_uer": False,
					"adaptations": [
						"native DGL one-layer DataLoader and SAGEModel execution",
						"index_copy_ backing-store writeback for retained intermediate rows",
						"final output publication into the shared query cache",
						"per-event affected-set reset and destination/self dependency coverage",
					],
				}
				if normalized_method == "rtec-uer"
				else None
			),
		},
		"environment": {
			"python": platform.python_version(),
			"torch": str(torch.__version__),
			"dgl": dgl.__version__,
			"threads": torch.get_num_threads(),
			"host": platform.node(),
			"device": str(device),
			"output_device": str(runner.outputs.device),
			"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
			"memory": {
				**measured_memory,
				"oracle_clone_logical_bytes": oracle_clone_bytes,
				"oracle_clone_physical_bytes": None,
				"initialization_snapshot_scope": "captured before independent oracle correctness check",
				"output_cache_bytes": runner.outputs.numel() * runner.outputs.element_size(),
				"retained_intermediate_state_bytes": (
					sum(value.numel() * value.element_size() for value in runner.states[1:])
					if isinstance(runner, UERRecompute) and runner.states is not None
					else 0
				),
				"retained_intermediate_state_total_referenced_bytes": (
					sum(value.numel() * value.element_size() for value in runner.states)
					if isinstance(runner, UERRecompute) and runner.states is not None
					else 0
				),
				"retained_input_feature_bytes_shared": (
					runner.states[0].numel() * runner.states[0].element_size()
					if isinstance(runner, UERRecompute) and runner.states is not None
					else 0
				),
				"declared_output_budget_bytes": output_budget_bytes,
				"temporary_intermediate_bytes_peak": (
					initialization_timing.get("temporary_intermediate_bytes_peak")
					if isinstance(initialization_timing, dict)
					else None
				),
			},
			"resource_assumptions": {
				"graph_and_features_resident_on_cpu": graph.device.type == "cpu" and features.device.type == "cpu",
				"one_native_block_and_feature_batch_on_model_device": True,
				"batch_size": batch_size,
				"publication": "not_applicable_standalone",
			},
		},
		"valid": True,
	}
	(output / "trial.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
	_phase(phase_file, "done", **phase_fields)
	return report
