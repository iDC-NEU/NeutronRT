"""Diagnose the frozen resident mutation path; run only within neu-sbox."""

import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer


def main(
	frozen_source: Annotated[Path, typer.Option(exists=True, file_okay=False)],
	output: Annotated[Path, typer.Option()],
	oracle: Annotated[bool, typer.Option("--oracle/--no-oracle")] = False,
	nested_fences: Annotated[bool, typer.Option("--nested-fences/--no-nested-fences")] = True,
):
	# Inspect the exact completed-cohort implementation while coding continues
	# independently in the main checkout. No current runtime module is imported.
	sys.path.insert(0, str(frozen_source.resolve()))
	import torch

	from benchmarks.streaming import decode_update
	from lib.dataset.inference import load_graph
	from lib.graph.resident_runtime import ResidentGraphSAGE
	from lib.profiling.workload import RecordedWorkload

	torch.set_num_threads(4)
	output.mkdir(parents=True, exist_ok=False)
	data = Path("/mnt/nfs0/lvhr/neutronrt-results/preparation/topology-five-20261116-v1/topologies/arxiv/graph.bin")
	trace = Path(
		"/mnt/nfs0/lvhr/neutronrt-results/matched-arxiv-gpu-20261117-v1/cohort/arxiv-NrtInc-r0/report/sage-NrtInc-0-workload.pt"
	)
	checkpoint = Path("/mnt/nfs0/lvhr/neutronrt-evaluation-data/checkpoints/sage-arxiv.pt")
	recording = RecordedWorkload.load(trace, checkpoint=checkpoint)
	graph = load_graph(data)
	features = graph.ndata.pop("feat")
	current: dict = {}
	rows = []

	def wrap(owner, attribute, label):
		original = getattr(owner, attribute)

		def measured(*args, **kwargs):
			if nested_fences:
				torch.cuda.synchronize()
			started = perf_counter()
			result = original(*args, **kwargs)
			if nested_fences:
				torch.cuda.synchronize()
			current[label] = current.get(label, 0.0) + perf_counter() - started
			return result

		setattr(owner, attribute, measured)

	def instrument(runtime):
		wrap(runtime, "_structure_delta", "structural_delta_preparation")
		wrap(runtime, "_admit_transition", "capacity_admission")
		wrap(runtime.graph, "add_edges", "graph_add_edges")
		wrap(runtime.graph, "remove_edges", "graph_remove_edges")
		wrap(torch.cuda, "mem_get_info", "device_free_memory_query_nested_in_admission")
		original = runtime.update

		def update(request):
			current.clear()
			result = original(request)
			rows.append(
				{
					"kind": result["kind"],
					"response_seconds": result["response_seconds"],
					"broad_phase_seconds": result["phase_seconds"],
					"nested_diagnostic_seconds": dict(current),
				}
			)
			return result

		runtime.update = update
		return runtime

	if oracle:
		spec = importlib.util.spec_from_file_location(
			"frozen_resident_driver", frozen_source / "scripts/evaluate/resident.py"
		)
		if spec is None or spec.loader is None:
			raise ImportError("Cannot load frozen resident driver")
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		module.__dict__["ResidentGraphSAGE"] = lambda *args, **kwargs: instrument(ResidentGraphSAGE(*args, **kwargs))
		module._run_nrt(
			graph,
			features,
			recording,
			output / "replay",
			device=torch.device("cuda:0"),
			capacity_bytes=8 * 1024**3,
			labels=None,
			test_mask=None,
		)
		rows = rows[-len(recording.steps) :]
	else:
		runtime = ResidentGraphSAGE(
			graph, features, recording.spec.build(), device="cuda:0", capacity_bytes=8 * 1024**3
		)
		runtime.bootstrap()
		for request in recording.warmup:
			runtime.update(decode_update(request))
		instrument(runtime)
		for request, _ in recording.steps:
			runtime.update(decode_update(request))
	value = {
		"scope": "Diagnostic with extra synchronization; nested times are not additive performance results",
		"frozen_source": str(frozen_source.resolve()),
		"interleaved_oracle": oracle,
		"nested_fences": nested_fences,
		"trace_digest": recording.digest,
		"rows": rows,
	}
	(output / "mutation-profile.json").write_text(json.dumps(value, indent=2) + "\n")
	print(json.dumps(value, indent=2))


if __name__ == "__main__":
	typer.run(main)
