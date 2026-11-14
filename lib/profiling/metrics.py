"""Inference latency, access, memory and communication measurements."""

from statistics import mean, stdev

from lib.graph.methods import validate_method
from lib.profiling.experiment import latency_summary

REFERENCE_URL = "https://arxiv.org/html/2603.20622v1"


def measurements(trial: dict) -> dict:
	profile = trial["profile"]
	workers = profile["workers"]
	updates = trial["updates"]
	latencies = [item["client_seconds"] for item in updates]
	edges = sum(item["changes"] for item in updates if item["kind"] in (2, 3))
	edge_only = all(item["kind"] in (2, 3) for item in updates)
	root = profile["coordinator"]

	def phase(report, prefix):
		return sum(value["total_seconds"] for key, value in report["timings"].items() if key.startswith(prefix))

	def counter(suffix):
		return sum(value for w in workers for key, value in w["counters"].items() if key.endswith(suffix))

	compiled = counter(".compiled_edges")
	delta_edges = counter(".routed_edges")
	actual = sum(
		w["operator_incremental"]["messages"]
		if w.get("operator_incremental") is not None
		else sum(value for key, value in w["counters"].items() if key.endswith((".compiled_edges", ".routed_edges")))
		for w in workers
	)
	retained = profile.get("retained_state") or {}
	return {
		"reference": REFERENCE_URL,
		"method": validate_method(trial["method"]),
		"fanout": trial["neighborhood_policy"]["fanout"],
		"response_time": {
			"mean_seconds": mean(latencies),
			"stddev_seconds": stdev(latencies) if len(latencies) > 1 else None,
			"max_seconds": max(latencies),
			"sum_seconds": sum(latencies),
			**latency_summary(latencies),
			"scope": "RPC submission to embedding publication, including Async+Wait; excludes query/trace generation",
		},
		"edge_updates_per_second": edges / sum(latencies) if edge_only else None,
		"edge_updates": edges,
		"workload_edge_only": edge_only,
		"accesses": {
			"compiled_neighbor_edges": compiled,
			"delta_routed_edges": delta_edges,
			"executed_edge_messages": actual,
			"source_vertex_rows": counter(".sources") + counter(".input_rows") + counter(".structural_rows"),
			"destination_vertex_rows": counter(".targets"),
			"reused_messages": sum((w.get("operator_incremental") or {}).get("reused", 0) for w in workers),
			"refresh_rows": sum((w.get("operator_incremental") or {}).get("refreshes", 0) for w in workers),
			"scope": "Measured-window accesses, including repeated tasks; messages include subtraction/addition work",
		},
		"memory": {
			"coordinator_peak_rss_bytes": root["rss"]["peak_rss_bytes"],
			"worker_peak_rss_bytes": [w["rss"]["peak_rss_bytes"] for w in workers],
			"gpu_peak_allocated_bytes": [(w.get("cuda") or {}).get("peak_allocated_bytes") for w in workers],
			"gpu_peak_reserved_bytes": [(w.get("cuda") or {}).get("peak_reserved_bytes") for w in workers],
			"resident_embeddings_bytes": [w["resident_current_bytes"] for w in workers],
			"historical_cpu_tensor_bytes": sum(
				(w.get("operator_incremental") or {}).get("history_bytes", 0) for w in workers
			),
			"published_replicas": retained,
			"scope": "Per-process RSS peaks; allocator metrics exclude driver/NCCL allocations",
		},
		"phases_seconds": {
			"graph_mutation": phase(root, "root.graph_insert") + phase(root, "root.graph_delete"),
			"graph_index_maintenance": phase(root, "root.graph_formats"),
			"affected_detection": phase(root, "root.affected_analysis"),
			"computation_graph_construction": phase(root, "root.block_compile") + phase(root, "root.dependency_plan"),
			"input_routing": phase(root, "root.input_routing"),
			"dispatch_and_wait": sum(
				phase(root, prefix)
				for prefix in (
					"root.task_dispatch_and_ready",
					"root.delta_dispatch_and_ready",
					"root.reconstruction_dispatch_and_ready",
				)
			),
			"worker_model_host": sum(
				phase(w, prefix)
				for w in workers
				for prefix in ("worker.model_cpu", "worker.delta_cpu", "worker.reconstruction_cpu")
			),
			"worker_model_cuda": sum(
				phase(w, prefix)
				for w in workers
				for prefix in ("worker.model_cuda", "worker.delta_cuda", "worker.reconstruction_cuda")
			),
			"scope": "Overlapping phase totals across ranks; not an additive wall-time breakdown",
		},
		"communication": {
			**profile["p2p"],
			"storage_selected_adjacency_bytes": sum(
				part["shared_response_bytes"] for part in (profile.get("storage") or {}).get("partitions", {}).values()
			),
			"scope": "Tensor payload bytes counted once at sender; excludes protocol overhead",
		},
		"embedding_mse": max(item["mean_squared_error"] for item in trial["correctness"]),
		"max_absolute_error": max(item["max_error"] for item in trial["correctness"]),
		"prediction_disagreements": sum(item["prediction_disagreements"] for item in trial["correctness"]),
		"classification_accuracy": {
			"value": None,
			"reason": "Requires trained checkpoint, labels and official test split",
		},
	}


def latency_envelope(trials: list[dict], bounds=(1, 5, 25, 50, 100)) -> list[dict]:
	"""Require all 20+ batches to meet the latency bound; report worst batch throughput."""
	result = []
	for bound in bounds:
		groups: dict[int, list[dict]] = {}
		for trial in trials:
			updates = trial["updates"]
			if updates:
				groups.setdefault(min(item["changes"] for item in updates), []).append(trial)
		eligible = [
			(size, trial)
			for size, group in groups.items()
			if all(
				t["valid"]
				and len(t["updates"]) >= 20
				and all(item["kind"] in (2, 3) and 0 < item["client_seconds"] <= bound for item in t["updates"])
				for t in group
			)
			for trial in group
		]
		if eligible:
			batch = max(item[0] for item in eligible)
			selected = [t for size, t in eligible if size == batch]
			throughput = min(item["changes"] / item["client_seconds"] for t in selected for item in t["updates"])
		else:
			batch, throughput = None, None
		result.append(
			{
				"latency_bound_seconds": bound,
				"largest_tested_batch": batch,
				"minimum_edge_updates_per_second": throughput,
				"scope": "Tested batch sizes only; not a global optimum",
			}
		)
	return result
