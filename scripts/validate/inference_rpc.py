"""Validate an external arXiv RPC server against independent full CPU inference."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Annotated, cast
from uuid import uuid4

import grpc
import torch
import typer

from lib.dataset.inference import load_graph as load_graph_data
from lib.models.spec import ModelSpec
from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceStub
from lib.rpc.limits import message_options
from lib.rpc.tensors import TensorPayload, decode_tensor, encode_tensor
from scripts.validate.inference import reference


def tensor(value):
	payload = encode_tensor(value)
	return wire.Tensor(dtype=cast(wire.Tensor.DType, payload.dtype), shape=payload.shape, data=payload.data)


def wait_ready(process, path, timeout=120):
	deadline = monotonic() + timeout
	while monotonic() < deadline:
		if process.poll() is not None:
			raise RuntimeError("Inference server exited before startup notification")
		if path.exists():
			value = json.loads(path.read_text())
			if value["pid"] == process.pid:
				return value
		sleep(min(0.05, max(0, deadline - monotonic())))
	else:
		raise TimeoutError("Inference server did not publish its startup identity")


def validate(name, args, folder):
	graph = load_graph_data(Path(args.data))
	features = graph.ndata.pop("feat")
	if features.shape != (169343, 128):
		raise ValueError("Expected official ogbn-arxiv graph/features")
	spec = ModelSpec(name, 128, 64, 40, checkpoint=str(args.checkpoint) if name == "sage" and args.checkpoint else None)
	model = spec.build()
	generator = torch.Generator().manual_seed(71)
	sources = (graph.out_degrees() > 0).nonzero().flatten()
	changed = sources[torch.randperm(sources.numel(), generator=generator)[:16]]
	frontier = graph.out_edges(graph.out_edges(changed)[1].unique())[1].unique()
	query = torch.cat([changed, frontier[:16], torch.randint(graph.num_nodes(), (128,), generator=generator)])[:128]
	ready_file = folder / f"{name}-{uuid4().hex}.ready.json"
	command = [
		sys.executable,
		"-m",
		"scripts.simple",
		"--data",
		str(args.data),
		"--workers",
		args.device,
		"--model",
		name,
		"--listen",
		"127.0.0.1:0",
		"--ready-file",
		str(ready_file),
		"--threads",
		str(args.threads),
		"--classes",
		str(spec.classes),
		"--hidden",
		str(spec.hidden),
		"--layers",
		str(spec.layers),
		"--heads",
		str(spec.heads),
	]
	if spec.checkpoint:
		command.extend(["--checkpoint", spec.checkpoint])
	log_path = folder / f"{name}.log"
	records = []
	with log_path.open("w") as log:
		process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		try:
			ready = wait_ready(process, ready_file)
			with grpc.insecure_channel(ready["address"], options=message_options()) as channel:
				grpc.channel_ready_future(channel).result(timeout=60)
				stub = OwnedInferenceStub(channel)
				session = stub.Info(wire.Empty(), timeout=10).session
				if session != ready["session"]:
					raise RuntimeError("RPC listener does not belong to the launched inference server")
				previous = None
				for version, operation in enumerate(("initialize", "features", "insert", "delete")):
					admission_seconds = completion_seconds = None
					completion = None
					if version:
						if operation == "features":
							first = changed
							second = features[changed] + torch.randn(16, 128, generator=generator) * 0.25
							kind = wire.Update.FEATURES
						elif operation == "insert":
							first = torch.randint(graph.num_nodes(), (16,), generator=generator)
							second, kind = query[:16], wire.Update.INSERT
						else:
							incoming = graph.in_edges(query.unique(), form="eid")
							eids = incoming[torch.randperm(incoming.numel(), generator=generator)[:16]]
							first, second = graph.find_edges(eids)
							kind = wire.Update.DELETE
						start = perf_counter()
						request = wire.Update(
							kind=kind,
							first=tensor(first),
							second=tensor(second),
							request_id=uuid4().hex,
							session=session,
						)
						if operation == "insert":
							ticket = stub.Async(request, timeout=10)
							admission_seconds = perf_counter() - start
							completion = stub.Wait(wire.WaitRequest(ticket=ticket), timeout=60)
						else:
							completion = stub.Sync(request, timeout=60)
						completion_seconds = perf_counter() - start
						assert completion.ready and completion.version == version
						if operation == "features":
							features.index_copy_(0, first, second)
						elif operation == "insert":
							graph.add_edges(first, second)
						else:
							_, _, removed = graph.edge_ids(first, second, return_uv=True)
							graph.remove_edges(removed.unique())
					start = perf_counter()
					response = stub.Query(wire.QueryRequest(nodes=tensor(query)), timeout=60)
					query_seconds = perf_counter() - start
					actual = decode_tensor(
						TensorPayload(response.values.dtype, tuple(response.values.shape), response.values.data)
					)
					expected = reference(model, graph, features, query)
					torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
					assert response.version == version
					changed_rows = (
						int(((expected - previous).abs().amax(1) > 1e-7).sum()) if previous is not None else 0
					)
					assert previous is None or changed_rows > 0
					info = stub.Info(wire.Empty(), timeout=10)
					assert info.completed_updates == version and info.completed_input_changes == version * 16
					record = {
						"operation": operation,
						"version": version,
						"max_abs_error": float((actual - expected).abs().max()),
						"prediction_disagreement": float((actual.argmax(1) != expected.argmax(1)).float().mean()),
						"changed_query_rows": changed_rows,
						"async_admission_seconds": admission_seconds,
						"update_completion_rtt_seconds": completion_seconds,
						"query_rtt_seconds": query_seconds,
						"affected_rows": list(completion.affected) if completion is not None else None,
					}
					records.append(record)
					print(json.dumps({"model": name, **record}), flush=True)
					previous = expected
					if completion is not None:
						stub.Release(completion.ticket, timeout=10)
		except BaseException:
			print(log_path.read_text(), file=sys.stderr)
			raise
		finally:
			if process.poll() is None:
				process.terminate()
			try:
				process.wait(timeout=20)
			except subprocess.TimeoutExpired:
				os.killpg(process.pid, signal.SIGKILL)
				process.wait()
		if process.returncode:
			raise RuntimeError(f"Inference server shutdown failed; inspect {log_path}")
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
	args.data = args.data.resolve()
	if args.checkpoint is not None:
		args.checkpoint = args.checkpoint.resolve()
	torch.set_num_threads(args.threads)
	with tempfile.TemporaryDirectory(prefix="neutronrt-rpc-arxiv-") as folder:
		report = {
			"dataset": "ogbn-arxiv",
			"device": args.device,
			"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
			"scope": "Separate RPC client/server processes and workers on one host; correctness on a shared GPU",
			"timing": "Client RTT includes encoding; full CPU oracle outside timing; no sustained throughput claim",
			"models": [validate(name, args, Path(folder)) for name in args.models],
		}
	args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
	app()
