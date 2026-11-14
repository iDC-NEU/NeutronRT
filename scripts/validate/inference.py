"""Validate process-owned updates against independent full CPU inference on arXiv."""

import json
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import torch
import torch.multiprocessing as mp
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.dist.comm_manager import CommManager
from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.ownership import Owner
from lib.models import GNNModel
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.runtime.worker import OwnedWorker


def build_model(name, args):
	torch.manual_seed(42)
	model = GNNModel(name, 128, 64, 40, 2, aggregator_type="mean", num_heads=2).eval()
	if name == "sage" and args.checkpoint:
		checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
		expected = {"model_name": "sage", "num_features": 128, "num_hiddens": 64, "num_classes": 40, "num_layers": 2}
		if any(checkpoint["model_config"][key] != value for key, value in expected.items()):
			raise ValueError("Expected the small two-layer arXiv GraphSAGE baseline checkpoint")
		model.load_state_dict(checkpoint["state_dict"])
	for layer in model.layers:
		if hasattr(layer, "set_allow_zero_in_degree"):
			layer.set_allow_zero_in_degree(True)
	return model


@torch.inference_mode()
def reference(model, graph, features, query):
	values = features
	for layer in range(model.num_layers):
		values = model(layer, graph, values)
	return values.index_select(0, query)


def _rank(rank, args, name, rendezvous, report_path):
	torch.set_num_threads(args.threads if rank == 0 else 1)
	devices = ("cpu", *args.workers)
	manager = CommManager.initialize(len(devices), rank, device=devices[rank], init_method=f"file://{rendezvous}")
	control = manager.create_epoch(range(len(devices)))
	model = build_model(name, args).to(devices[rank])
	if rank:
		OwnedWorker(manager, model, Owner(rank, 1), max_bytes=args.worker_bytes, control_epoch=control).run()
		return
	transport = OwnedTransport(
		manager,
		control,
		[
			RemoteEndpoint(Owner(peer, 1), torch.device(devices[peer]), args.worker_bytes)
			for peer in range(1, len(devices))
		],
	)
	plane = RemoteEmbeddingPlane(transport, storage_bytes=args.storage_bytes)
	admission = None
	try:
		load_start = perf_counter()
		graph = load_graph_data(Path(args.data))
		features = graph.ndata.pop("feat")
		if features.shape != (169343, 128):
			raise ValueError("This validation expects the official ogbn-arxiv graph/features")
		graph.create_formats_()
		service = DistributedEmbeddingService(
			graph,
			features,
			(128, 64, 40),
			plane,
			batch_size=args.batch_size,
			pipeline=True,
			source_degree_sensitive=name == "gcn",
		)
		admission = UpdateAdmission(service)
		generator = torch.Generator().manual_seed(71)
		sources = (graph.out_degrees() > 0).nonzero().flatten()
		changed = sources[torch.randperm(sources.numel(), generator=generator)[: args.changes]]
		frontier = changed
		for _ in range(model.num_layers):
			frontier = graph.out_edges(frontier)[1].unique()
			low_degree = frontier[graph.in_degrees(frontier) <= 200]
			if low_degree.numel():
				frontier = low_degree
		query = torch.cat(
			[
				changed,
				frontier[: args.changes],
				torch.randint(graph.num_nodes(), (args.query_size,), generator=generator),
			]
		)[: args.query_size]
		pointer = features.data_ptr()
		records = []
		previous = None
		for operation in ("initialize", "features", "insert", "delete"):
			start = perf_counter()
			result = None
			if operation == "initialize":
				service.ensure(query)
			elif operation == "features":
				values = features[changed] + torch.randn(args.changes, 128, generator=generator) * 0.25
				result = admission.sync(UpdateKind.FEATURES, changed, values, borrow=True)
			elif operation == "insert":
				u = torch.randint(graph.num_nodes(), (args.changes,), generator=generator)
				v = query[: args.changes]
				result = admission.submit(UpdateKind.INSERT, u, v, borrow=True).result()
			else:
				incoming = graph.in_edges(query.unique(), form="eid")
				eids = incoming[torch.randperm(incoming.numel(), generator=generator)[: args.changes]]
				u, v = graph.find_edges(eids)
				result = admission.sync(UpdateKind.DELETE, u, v, borrow=True)
			publication_seconds = perf_counter() - start
			resident_before_read = plane.storage.store.bytes_used
			with service.query(query, plane.storage.owner) as actual:
				expected = reference(model, graph, features, query)
				torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
				changed_rows = int(((expected - previous).abs().amax(1) > 1e-7).sum()) if previous is not None else 0
				if previous is not None and not changed_rows:
					raise AssertionError("The update trace did not change a queried embedding")
				record = {
					"operation": operation,
					"version": service.catalog.version,
					"publication_seconds": publication_seconds,
					"max_abs_error": float((actual - expected).abs().max()),
					"prediction_disagreement": float((actual.argmax(1) != expected.argmax(1)).float().mean()),
					"changed_query_rows": changed_rows,
					"query_rows": query.numel(),
					"storage_bytes_before_query": resident_before_read,
					"storage_bytes_after_query": plane.storage.store.bytes_used,
					"worker_resident_bytes": {
						node.owner.rank: node.bytes_used for node in transport.endpoints.values()
					},
					"affected_rows": result.affected if result else None,
				}
				previous = expected
			assert features.data_ptr() == pointer
			assert not torch.cuda.is_initialized()
			records.append(record)
			print(json.dumps({"model": name, **record}), flush=True)
		Path(report_path).write_text(
			json.dumps(
				{
					"model": name,
					"weights": "trained baseline" if name == "sage" and args.checkpoint else "fixed random",
					"nodes": graph.num_nodes(),
					"edges_after_updates": graph.num_edges(),
					"load_and_validation_seconds": perf_counter() - load_start,
					"completed_updates": service.counters.updates,
					"completed_input_changes": service.counters.input_changes,
					"records": records,
				},
				indent=2,
			)
			+ "\n"
		)
	finally:
		if admission is not None:
			admission.close()
		plane.close()
		transport.stop()
		manager.destroy()


ModelsChoice = StrEnum(  # type: ignore[misc]  # Mypy requires literal Enum members.
	"ModelsChoice", {str(value).replace("-", "_").upper(): value for value in ["sage", "gat", "gcn"]}
)


@dataclass
class Options:
	data: Path
	checkpoint: Path | None
	workers: list[str]
	models: list[str]
	changes: int
	query_size: int
	batch_size: int
	threads: int
	worker_bytes: int
	storage_bytes: int
	output: Path


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	data: Annotated[Path, typer.Option("--data")],
	output: Annotated[Path, typer.Option("--output")],
	checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
	workers: Annotated[list[str] | None, typer.Option("--workers")] = None,
	models: Annotated[list[ModelsChoice] | None, typer.Option("--models")] = None,
	changes: Annotated[int, typer.Option("--changes")] = 16,
	query_size: Annotated[int, typer.Option("--query-size")] = 128,
	batch_size: Annotated[int, typer.Option("--batch-size")] = 8192,
	threads: Annotated[int, typer.Option("--threads")] = 4,
	worker_bytes: Annotated[int, typer.Option("--worker-bytes")] = 1024**3,
	storage_bytes: Annotated[int, typer.Option("--storage-bytes")] = 2 * 1024**3,
):
	args = Options(
		**{
			**locals(),
			"workers": workers if workers is not None else ["cuda:0", "cuda:1"],
			"models": [item.value for item in models] if models is not None else ["sage", "gat"],
		}
	)
	if min(args.changes, args.batch_size, args.threads, args.worker_bytes, args.storage_bytes) < 1:
		raise typer.BadParameter("Counts and budgets must be positive")
	if args.query_size < args.changes * 2:
		raise typer.BadParameter("query-size must cover changed vertices and their downstream targets")
	models = []
	with tempfile.TemporaryDirectory(prefix="neutronrt-owned-arxiv-") as folder:
		for name in args.models:
			report = Path(folder) / f"{name}.json"
			mp.spawn(
				_rank,
				args=(args, name, str(Path(folder) / f"{name}-store"), str(report)),
				nprocs=len(args.workers) + 1,
				join=True,
			)
			models.append(json.loads(report.read_text()))
	args.output.write_text(
		json.dumps(
			{
				"dataset": "ogbn-arxiv",
				"workers": args.workers,
				"batch_size": args.batch_size,
				"scope": "Process-owned inference on one host; CPU coordinator, real P2P; excludes RPC latency",
				"oracle": "Independent full CPU inference outside publication timing; no CPU serving embedding matrix",
				"models": models,
			},
			indent=2,
		)
		+ "\n"
	)


if __name__ == "__main__":
	app()
