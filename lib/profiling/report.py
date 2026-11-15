"""Readable comparisons backed by retained raw experiment windows."""

from statistics import median

from lib.graph.methods import method_from_switches


def markdown_report(report: dict) -> str:
	rows = []
	comparisons = {item["case"]: item["verdict"] for item in report["comparisons"]}
	for case in dict.fromkeys(run["case"] for run in report["runs"]):
		all_runs = [run for run in report["runs"] if run["case"] == case]
		runs = [run for run in all_runs if run["valid"]]

		def typical(values, scale=1, precision=2):
			valid = [value * scale for value in values if value is not None]
			return f"{median(valid):.{precision}f}" if valid else "—"

		query_errors = [sample["max_abs_error"] for run in runs for sample in run.get("query_accuracy", [])]
		accuracy_losses = [
			sample["accuracy_loss_percentage_points"]
			for run in runs
			for sample in run.get("query_accuracy", [])
			if sample.get("accuracy_loss_percentage_points") is not None
		]
		verdict = comparisons.get(case, "reference") if len(runs) == len(all_runs) else "invalid"
		rows.append(
			"| "
			+ " | ".join(
				[
					case,
					f"{len(runs)}/{len(all_runs)}",
					typical([run["client_changes_per_second"] for run in runs]),
					typical([run["profile"]["engine"]["completed_changes_per_second"] for run in runs]),
					typical([run["profile"]["engine"]["rows"] for run in runs], precision=0),
					typical(
						[
							sum(
								value
								for worker in run["profile"]["workers"]
								for name, value in worker["counters"].items()
								if name.startswith("model.layer") and name.endswith(".compiled_edges")
							)
							for run in runs
						],
						precision=0,
					),
					typical([run["profile"]["engine"].get("delta_routed_edges", 0) for run in runs], precision=0),
					typical([run["profile"]["engine"].get("delta_refresh_rows", 0) for run in runs], precision=0),
					typical([run["profile"]["p2p"]["send_payload_bytes"] for run in runs], scale=1 / 1024**2),
					typical(
						[(run["profile"].get("network") or {}).get("totals", {}).get("bytes_sent") for run in runs],
						scale=1 / 1024**2,
					),
					typical([run["query_latency"]["p99_seconds"] for run in runs], scale=1000),
					f"{max(query_errors):.3g}" if query_errors else "—",
					f"{max(accuracy_losses):.3f}" if accuracy_losses else "—",
					verdict,
					all_runs[0].get("method")
					or (
						method_from_switches(all_runs[0]["enabled_switches"])
						if "enabled_switches" in all_runs[0]
						else "—"
					),
				]
			)
			+ " |"
		)
	return "\n".join(
		[
			"Matched inference experiments. Values are medians of valid windows; inspect valid/total counts.",
			"Failed trials stay in report.json and invalidate comparisons. Verdicts describe this host and workload.",
			"",
			"| Case | Valid | Client changes/s | Engine changes/s | Model rows | "
			"Full-neighbor edges | Delta edges | Refresh rows | "
			"P2P send MiB | NIC sent MiB | Query p99 ms | Max query error | Worst accuracy loss pp | "
			"Verdict | Method |",
			"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
			*rows,
			"",
			(
				"Accuracy covers the full official test split at each observed graph version."
				if report.get("full_test_coverage")
				else "Accuracy uses captured selected test rows, rather than the entire official test split."
			),
			"P2P bytes are application tensor payloads. NIC bytes are host interface counters and may include "
			"unrelated traffic.",
			"CPU/CUDA phases overlap. Detailed traces run separately from timings. Raw reports retain all samples.",
			"",
		]
	)
