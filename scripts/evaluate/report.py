"""Aggregate immutable distributed trials into comparable JSON and CSV metrics."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Annotated

import typer

from lib.graph.methods import METHODS, NRT_INC, RTEC_FULL, RTEC_NS, validate_method
from lib.profiling.experiment import paired_comparison
from lib.profiling.metrics import REFERENCE_URL, latency_envelope


def median_mib(rows: list[dict], name: str) -> str:
	values = [row[name] for row in rows if row[name] is not None]
	return f"{median(values) / 1024**2:.2f}" if values else "—"


def aggregate(inputs: list[Path]) -> dict:
	if len({path.name for path in inputs}) != len(inputs):
		raise ValueError("Input directories must have distinct names for unambiguous cohort labels")
	runs, comparisons, envelopes, rows = [], [], [], []
	latency_groups = defaultdict(list)
	for path in inputs:
		summary = json.loads((path / "summary.json").read_text())
		groups = defaultdict(list)
		for item in summary["trials"]:
			trial = json.loads((path / item["file"]).read_text())
			metrics = trial["metrics"]
			method = validate_method(trial["method"])
			metrics.update(method=method, reference=REFERENCE_URL)
			runs.append(
				{
					"file": item["file"],
					"cohort": path.name,
					"valid": trial["valid"],
					"metrics": metrics,
					"method": method,
					"rpc": trial.get("rpc"),
				}
			)
			groups[trial["model"], trial["method"]].append(trial)
			fanout = trial["neighborhood_policy"]["fanout"]
			latency_groups[trial["identity"], trial["world_size"], trial["model"], trial["method"], fanout].append(
				trial
			)
			rows.append(
				{
					"cohort": path.name,
					"model": trial["model"],
					"method": metrics["method"],
					"fanout": fanout,
					"repetition": trial["repetition"],
					"valid": trial["valid"],
					"nodes": trial["nodes"],
					"edges": trial["edges"],
					"layers": trial["spec"]["layers"],
					"world_size": trial["world_size"],
					"response_mean_seconds": metrics["response_time"]["mean_seconds"],
					"response_p99_seconds": metrics["response_time"]["p99_seconds"],
					"edge_updates_per_second": metrics["edge_updates_per_second"],
					**metrics["accesses"],
					"cpu_coordinator_peak_bytes": metrics["memory"]["coordinator_peak_rss_bytes"],
					"gpu_peak_allocated_sum_bytes": sum(v or 0 for v in metrics["memory"]["gpu_peak_allocated_bytes"]),
					"history_cpu_bytes": metrics["memory"]["historical_cpu_tensor_bytes"],
					"p2p_payload_bytes": metrics["communication"]["send_payload_bytes"],
					"network_bytes_sent": (metrics["communication"].get("network") or {})
					.get("totals", {})
					.get("bytes_sent"),
					"network_bytes_recv": (metrics["communication"].get("network") or {})
					.get("totals", {})
					.get("bytes_recv"),
					"embedding_mse": metrics["embedding_mse"],
					"prediction_disagreement_fraction": sum(c["prediction_disagreements"] for c in trial["correctness"])
					/ sum(c["rows"] for c in trial["correctness"]),
				}
			)
		for (model, method), trials in groups.items():
			if method == NRT_INC or (model, NRT_INC) not in groups:
				continue
			baseline, candidate = trials, groups[model, NRT_INC]
			comparison = paired_comparison(baseline, candidate, metric="response_seconds")

			def total(group, name):
				return mean(t["metrics"]["accesses"][name] for t in group)

			base_edges = total(baseline, "executed_edge_messages")
			comparisons.append(
				{
					"cohort": path.name,
					"model": model,
					"baseline": method,
					"candidate": NRT_INC,
					"fanout": baseline[0]["neighborhood_policy"]["fanout"],
					**comparison,
					"message_reduction_fraction": 1 - total(candidate, "executed_edge_messages") / base_edges
					if base_edges
					else None,
				}
			)
	for (identity, world, model, method, fanout), trials in latency_groups.items():
		envelopes.append(
			{
				"identity": identity,
				"world_size": world,
				"model": model,
				"method": method,
				"fanout": fanout,
				"bounds": latency_envelope(trials),
			}
		)
	return {
		"schema_version": 2,
		"reference": REFERENCE_URL,
		"runs": runs,
		"comparisons": comparisons,
		"latency_envelopes": envelopes,
		"rows": rows,
		"coverage": {
			"response_time_and_throughput_fig7_fig10": "measured",
			"vertex_edge_accesses_fig8_fig11": "measured with distinct compiled-row and executed-message scopes",
			"gpu_cpu_memory_fig9_table6": "partial: allocator/RSS/logical tensors; no Nsight component attribution",
			"runtime_breakdown_fig10": "partial: overlapping phase totals, not additive wall-time percentages",
			"batch_size_latency_bounds_fig12abc": "measured only across supplied cohorts and tested sizes",
			"odec_query_sensitivity_fig12d": "not measured: ready reads do not exercise ODEC recomputation",
			"degree_percentile_access_reduction_table5": "not measured: no degree attribution",
			"layer_sensitivity_table7": "measured only if supplied cohorts vary layers",
			"accuracy_table4": "embedding MSE measured; trained classification/retraining baselines not measured",
			"external_baselines_fig12e": "not measured: InkStream/Ripple implementations not available",
			"baselines": sorted({row["method"] for row in rows}),
			"billion_edge_out_of_memory": "not established by supplied completed trials",
		},
	}


def main(
	inputs: Annotated[list[Path], typer.Argument(help="Result directories containing summary.json")],
	output: Annotated[Path, typer.Option(help="New directory; existing output is never overwritten")],
):
	report = aggregate(inputs)
	output.mkdir(parents=True, exist_ok=False)
	with (output / "metrics.json").open("x") as stream:
		json.dump(report, stream, indent=2, allow_nan=False)
	with (output / "metrics.csv").open("x", newline="") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
		writer.writeheader()
		writer.writerows(report["rows"])
	with (output / "comparison.md").open("x") as stream:
		stream.write(comparison_table(report))
	typer.echo(str(output))


def comparison_table(report: dict) -> str:
	"""A compact latency table; detailed accuracy and resource columns stay in CSV."""
	rows = report["rows"]
	lines = [
		"Mean update response time (ms), median across repetitions.",
		"",
		"| Cohort | Model | Sampling fanout | " + " | ".join(METHODS) + " |",
		"| --- | --- | ---: | " + " | ".join(["---:"] * len(METHODS)) + " |",
	]
	groups = list(dict.fromkeys((row["cohort"], row["model"]) for row in rows))
	for cohort, model in groups:
		fanouts = {
			row["fanout"]
			for row in rows
			if row["cohort"] == cohort and row["model"] == model and row["method"] == RTEC_NS
		}
		if len(fanouts) > 1:
			raise ValueError("Separate sampling fanouts into distinct experiment cohorts")
		fanout = str(next(iter(fanouts))) if fanouts else "—"
		values = []
		for method in METHODS:
			group = [r for r in rows if r["cohort"] == cohort and r["model"] == model and r["method"] == method]
			valid = [r for r in group if r["valid"]]
			values.append(
				f"{median(r['response_mean_seconds'] for r in valid) * 1000:.2f}"
				if valid and len(valid) == len(group)
				else "—"
			)
		lines.append(f"| {cohort} | {model} | {fanout} | " + " | ".join(values) + " |")
	lines.extend(
		[
			"",
			"RTEC-Full recomputes complete neighborhoods of affected outputs; it is not full-graph recomputation.",
			"RTEC-NS uses its configured fanout at every layer and is approximate. "
			"Sampling error is reported separately.",
			"",
			"| Cohort | Model | Sampling method | Max embedding MSE | Prediction disagreement (%) |",
			"| --- | --- | --- | ---: | ---: |",
		]
	)
	for cohort, model in groups:
		for method in METHODS:
			if method in (RTEC_FULL, NRT_INC, "RTEC-UER"):
				continue
			group = [
				r
				for r in rows
				if r["cohort"] == cohort and r["model"] == model and r["method"] == method and r["valid"]
			]
			if group:
				lines.append(
					f"| {cohort} | {model} | {method} | {max(r['embedding_mse'] for r in group):.4g} | "
					f"{median(r['prediction_disagreement_fraction'] for r in group) * 100:.2f} |"
				)
	lines.extend(
		[
			"",
			"Measured communication counters are reported separately from application payloads.",
			"NIC counters are host-interface totals and may include unrelated traffic; missing counters remain `—`.",
			"",
			"| Cohort | Model | Method | P2P payload (MiB) | NIC sent (MiB) | NIC received (MiB) |",
			"| --- | --- | --- | ---: | ---: | ---: |",
		]
	)
	for cohort, model in groups:
		for method in METHODS:
			group = [
				r
				for r in rows
				if r["cohort"] == cohort and r["model"] == model and r["method"] == method and r["valid"]
			]
			if not group:
				continue

			lines.append(
				f"| {cohort} | {model} | {method} | {median_mib(group, 'p2p_payload_bytes')} | "
				f"{median_mib(group, 'network_bytes_sent')} | {median_mib(group, 'network_bytes_recv')} |"
			)
	return "\n".join(lines) + "\n"


if __name__ == "__main__":
	typer.run(main)
