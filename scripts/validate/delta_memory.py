"""Stress sparse-update workspace admission under a restricted CUDA allocator."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import dgl
import torch
import typer

from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.profiling.experiment import source_fingerprint
from lib.runtime.config import RuntimeFeatures
from lib.runtime.launch import Placement, WorkerSettings, run_worker
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import RemoteEndpoint

app = typer.Typer(help=__doc__, pretty_exceptions_show_locals=False)


@app.command()
def main(
	output: Annotated[Path, typer.Option(help="New directory for the source snapshot and validation report")],
	nodes: Annotated[int, typer.Option(min=1)] = 300_000,
	features: Annotated[int, typer.Option(min=1)] = 128,
	hidden: Annotated[int, typer.Option(min=1)] = 128,
	classes: Annotated[int, typer.Option(min=1)] = 64,
	worker_bytes: Annotated[int, typer.Option(min=1)] = 512 * 1024**2,
	storage_bytes: Annotated[int, typer.Option(min=1)] = 2 * 1024**3,
	anchor_bytes: Annotated[int, typer.Option(min=1)] = 2 * 1024**3,
	cuda_memory_fraction: Annotated[float, typer.Option(min=0.001, max=1)] = 0.04,
	initialization_batch_size: Annotated[int, typer.Option(min=1)] = 65536,
	threads: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	torch.set_num_threads(threads)
	output = output.resolve()
	rendezvous = output.with_suffix(".store").as_uri()
	placement = Placement.resolve(init_method=rendezvous)
	if placement.rank is None or placement.world_size != 2:
		raise typer.BadParameter("Launch this GPU gate with torchrun --standalone --nproc-per-node=2")
	switches = RuntimeFeatures(
		profiling=True,
		retain_aggregates=True,
		incremental_aggregates=True,
		row_refresh=True,
		gather_inputs=True,
		coalescing=True,
	)
	settings = WorkerSettings(switches, threads=threads, timeout=120, rendezvous_prefix="delta-memory-stress")
	if placement.rank:
		torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, 0)
		run_worker(None, RemoteEndpoint(Owner(1, 1), torch.device("cuda:0"), worker_bytes), 2, rendezvous, settings)
		return
	output.mkdir(parents=True, exist_ok=False)
	fingerprint = source_fingerprint(output / "sources.zip")
	vertices = torch.arange(nodes)
	graph = dgl.graph(
		(torch.cat((torch.zeros(nodes, dtype=torch.int64), vertices)), vertices.repeat(2)), num_nodes=nodes
	)
	values = torch.randn(nodes, features, generator=torch.Generator().manual_seed(71)) * 0.1
	spec = ModelSpec("sage", features, hidden, classes, layers=2)
	model = spec.build()
	with ManagedRuntime(
		graph,
		values,
		spec,
		placement=placement,
		switches=switches,
		worker_threads=threads,
		worker_bytes=worker_bytes,
		storage_bytes=storage_bytes,
		delta_anchor_bytes=anchor_bytes,
		batch_size=1048576,
		max_plan_edges=max(2_000_000, graph.num_edges()),
		timeout=120,
		rendezvous_prefix=settings.rendezvous_prefix,
	) as runtime:
		for layer in (1, 2):
			for start in range(0, nodes, initialization_batch_size):
				runtime.service.ensure(torch.arange(start, min(nodes, start + initialization_batch_size)), layer)
			logging.info("Initialized layer %d over %d vertices", layer, nodes)
		sample = torch.cat((torch.tensor([0]), torch.arange(1, nodes, max(1, nodes // 256))))
		checks = []

		def check():
			with torch.inference_mode():
				expected = model(1, graph, model(0, graph, values))[sample]
				with runtime.service.query(sample, runtime.plane.storage.owner) as actual:
					torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
					checks.append(float((actual - expected).abs().max()))

		check()
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			for value in (0.25, -0.1):
				runtime.service.change_features(torch.tensor([0]), torch.full((1, features), value))
			report = profiler.stop()
		check()
		if not report["valid"] or torch.cuda.is_initialized():
			raise AssertionError("Invalid profiling window or CUDA initialized in the CPU coordinator")
		result = {
			"valid": True,
			"scope": "synthetic CUDA workspace stress; no performance claim",
			"source_fingerprint": fingerprint,
			"nodes": nodes,
			"spec": asdict(spec),
			"worker_resident_budget_bytes": worker_bytes,
			"torch_cuda_memory_fraction": cuda_memory_fraction,
			"max_errors": checks,
			"counters": asdict(runtime.service.counters),
			"profile": report,
		}
		(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
	logging.info("Passed bounded CUDA workspace stress")


if __name__ == "__main__":
	app()
