"""Inject worker exits on arXiv and compare recovered results with full inference."""

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Annotated

import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.data_plane import RejectedPublication
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from scripts.validate.inference import reference


def terminate(runtime, owner):
	process = runtime.processes[owner.rank]
	process.kill()
	process.join(5)
	if process.is_alive():
		raise RuntimeError("Injected worker exit did not complete")


def validate(name, args):
	graph = load_graph_data(Path(args.data))
	features = graph.ndata.pop("feat")
	if features.shape != (169343, 128):
		raise ValueError("Expected official ogbn-arxiv graph/features")
	graph.create_formats_()
	pointer = features.data_ptr()
	spec = ModelSpec(name, 128, 64, 40, checkpoint=str(args.checkpoint) if name == "sage" and args.checkpoint else None)
	model = spec.build()
	generator = torch.Generator().manual_seed(71)
	sources = (graph.out_degrees() > 0).nonzero().flatten()
	changed = sources[torch.randperm(sources.numel(), generator=generator)[:16]]
	frontier = graph.out_edges(graph.out_edges(changed)[1].unique())[1].unique()
	query = torch.cat([changed, frontier[:64], torch.randint(graph.num_nodes(), (128,), generator=generator)])[:128]
	records = []
	with ManagedRuntime(
		graph,
		features,
		spec,
		[args.device],
		timeout=30,
		batch_size=16,
		switches=RuntimeFeatures(pipeline=True),
		worker_bytes=1024**3,
		storage_bytes=256 * 1024**2,
	) as runtime:
		service, plane = runtime.service, runtime.service.remote_plane
		service.ensure(query)
		for scenario in ("unflushed", "before_writeback_ack", "partial_update", "acknowledged_replica"):
			groups = service.catalog.selections(2, query, service.catalog.version)
			key = groups[0].key
			producer = plane.directory.locate(key).owner
			before_tasks, before_rows = service.counters.tasks, service.counters.recovery_rows
			before_seconds, before_resets = service.counters.recovery_seconds, runtime.reset_count
			start = perf_counter()
			if scenario == "partial_update":
				compute, layer_two = plane.transport.compute, 0

				def interrupted(owner, packet, compute=compute, **kwargs):
					nonlocal layer_two
					if packet.header.layer == 2:
						layer_two += 1
						if layer_two == 2:
							terminate(runtime, owner)
					return compute(owner, packet, **kwargs)

				plane.transport.compute = interrupted
				edges = graph.num_edges()
				admission = UpdateAdmission(service)
				try:
					result = admission.sync(UpdateKind.INSERT, changed, query[:16], borrow=True)
				finally:
					admission.close()
				assert layer_two == 2 and result.version == 1
				assert graph.num_edges() == edges + 16 and service.counters.updates == 1
				assert service.counters.input_changes == 16 and service.resume_update(1) is result
			elif scenario == "before_writeback_ack":
				started, proceed = Event(), Event()
				complete = plane.directory.complete_copy

				def delayed_ack(attempt, handle, started=started, proceed=proceed, complete=complete):
					started.set()
					if not proceed.wait(60):
						raise TimeoutError("Recovery did not release the writeback test gate")
					return complete(attempt, handle)

				plane.directory.complete_copy = delayed_ack
				pending = plane.writeback(key, plane.storage.owner)
				try:
					assert started.wait(15)
					terminate(runtime, producer)
					with service.query(query, plane.storage.owner):
						pass
				finally:
					proceed.set()
					plane.directory.complete_copy = complete
				try:
					pending.result(timeout=15)
				except RejectedPublication:
					pass
				else:
					raise AssertionError("A writeback from the lost producer was accepted")
			else:
				if scenario == "acknowledged_replica":
					for group in groups:
						plane.writeback(group.key, plane.storage.owner).result(timeout=15)
				terminate(runtime, producer)
			with service.query(query, plane.storage.owner) as actual:
				recovered = actual.clone()
			seconds = perf_counter() - start
			expected = reference(model, graph, features, query)
			torch.testing.assert_close(recovered, expected, rtol=2e-4, atol=3e-5)
			assert runtime.reset_count == before_resets + 1
			assert features.data_ptr() == pointer and not torch.cuda.is_initialized()
			if scenario == "acknowledged_replica":
				assert service.counters.tasks == before_tasks
			else:
				assert service.counters.recovery_rows > before_rows and plane.storage.store.bytes_used == 0
			record = {
				"scenario": scenario,
				"graph_version": service.catalog.version,
				"max_abs_error": float((recovered - expected).abs().max()),
				"prediction_disagreement": float((recovered.argmax(1) != expected.argmax(1)).float().mean()),
				"worker_generation": next(iter(plane.remotes)).generation,
				"scenario_seconds": seconds,
				"recovery_seconds": service.counters.recovery_seconds - before_seconds,
				"recovery_rows": service.counters.recovery_rows - before_rows,
				"model_tasks": service.counters.tasks - before_tasks,
				"cpu_replica_bytes": plane.storage.store.bytes_used,
			}
			records.append(record)
			print(json.dumps({"model": name, **record}), flush=True)
	return {"model": name, "weights": "trained baseline" if spec.checkpoint else "fixed random", "records": records}


ModelsChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"ModelsChoice", {str(value).replace("-", "_").upper(): value for value in ["sage", "gat", "gcn"]}
)


@dataclass
class Options:
	data: Path
	checkpoint: Path | None
	device: str
	models: list[str]
	threads: int
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	device: Annotated[str, typer.Option("--device")] = "cuda:0",
	models: Annotated[list[ModelsChoice] | None, typer.Option("--models")] = None,
	threads: Annotated[int, typer.Option("--threads")] = 4,
):
	args = Options(
		**{**locals(), "models": [item.value for item in models] if models is not None else ["sage", "gat", "gcn"]}
	)
	if args.threads < 1:
		raise typer.BadParameter("threads must be positive")
	torch.set_num_threads(args.threads)
	report = {
		"dataset": "ogbn-arxiv",
		"device": args.device,
		"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
		"scope": "Actual managed worker exits; master stays live; no RPC or steady-state performance claim",
		"oracle": "Independent full CPU inference after each recovery, outside scenario timing",
		"models": [validate(name, args) for name in args.models],
	}
	args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	app()
