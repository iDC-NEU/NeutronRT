"""Measure existing gRPC interfaces from a separate client process."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import grpc
import torch
import typer

from lib.rpc import inference_pb2 as wire
from lib.rpc.inference_pb2_grpc import OwnedInferenceStub
from lib.rpc.limits import message_options
from lib.rpc.service import _from_wire, _to_wire
from lib.rpc.timing import RpcTimings

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def probe(
	address: Annotated[str, typer.Option(help="Existing inference server host:port")],
	output: Annotated[Path, typer.Option(help="New JSON file; refuses to overwrite")],
	requests: Annotated[int, typer.Option(min=1)] = 100,
	warmup: Annotated[int, typer.Option(min=0)] = 10,
	query_sizes: Annotated[list[int] | None, typer.Option(help="Repeat for each query size")] = None,
	timeout: Annotated[float, typer.Option(min=0.001)] = 120,
	concurrency: Annotated[int, typer.Option(min=1, max=256)] = 1,
	updates: Annotated[
		int, typer.Option(min=0, help="Optional sequential feature mutations through Sync/Async+Wait")
	] = 0,
	changes: Annotated[int, typer.Option(min=1)] = 1,
	seed: int = 71,
):
	"""Record Info RTT and query API latencies over the actual client connection."""
	if output.exists():
		raise typer.BadParameter("Output already exists; choose a new path to preserve previous measurements")
	timings = RpcTimings()
	sizes = query_sizes or [1, 64, 1024]
	if min(sizes) < 1:
		raise typer.BadParameter("Query sizes must be positive")
	with grpc.insecure_channel(address, options=message_options()) as channel:
		grpc.channel_ready_future(channel).result(timeout=timeout)
		measured = grpc.intercept_channel(channel, timings)
		stub = OwnedInferenceStub(measured)

		def info_call():
			return stub.Info(wire.Empty(), timeout=timeout)

		node_count = info_call().nodes

		def query(nodes):
			return stub.Query(wire.QueryRequest(nodes=_to_wire(nodes)), timeout=timeout)

		for _ in range(warmup):
			info_call()
		# Warm each query shape before measuring; this command measures ready reads.
		for size in sizes:
			query(torch.arange(min(size, node_count)))

		def measure(call):
			timings.clear()
			started = perf_counter()
			if concurrency == 1:
				for _ in range(requests):
					call()
			else:
				with ThreadPoolExecutor(max_workers=concurrency) as executor:
					list(executor.map(lambda _: call(), range(requests)))
			elapsed = perf_counter() - started
			return {**timings.report(), "wall_seconds": elapsed, "requests_per_second": requests / elapsed}

		info_report = measure(info_call)
		queries = []
		for size in sizes:
			nodes = torch.arange(min(size, node_count))
			queries.append({"nodes": min(size, node_count), **measure(lambda nodes=nodes: query(nodes))})
		update_report = None
		if updates:
			generator = torch.Generator().manual_seed(seed)
			nodes = torch.randperm(node_count, generator=generator)[:changes]
			response = stub.Query(wire.QueryRequest(nodes=_to_wire(nodes), layer=0), timeout=timeout)
			base = _from_wire(response.values, 64 * 1024**2)
			prefix = uuid4().hex
			timings.clear()
			completion_seconds = []
			for index in range(updates):
				values = base + torch.randn(base.shape, generator=generator) * (index + 1) * 0.01
				request = wire.Update(
					kind=wire.Update.FEATURES,
					first=_to_wire(nodes),
					second=_to_wire(values),
					request_id=f"{prefix}-{index}",
				)
				started = perf_counter()
				if index % 2:
					ticket = stub.Async(request, timeout=timeout)
					completion = stub.Wait(wire.WaitRequest(ticket=ticket), timeout=timeout)
				else:
					completion = stub.Sync(request, timeout=timeout)
				if not completion.ready:
					raise RuntimeError("Update did not publish its embeddings")
				completion_seconds.append(perf_counter() - started)
				stub.Release(completion.ticket, timeout=timeout)
			update_report = {
				**timings.report(),
				"completion_seconds": completion_seconds,
				"changes_per_batch": nodes.numel(),
				"kind": "features",
			}
	result = {
		"address": address,
		"graph_nodes": node_count,
		"api": "inference",
		"concurrency": concurrency,
		"info": info_report,
		"queries": queries,
		"updates": update_report,
		"scope": "Separate-process client; Info is application RTT, Query reads warmed embeddings, not ODEC",
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("x") as stream:
		json.dump(result, stream, indent=2, allow_nan=False)
	typer.echo(str(output))


if __name__ == "__main__":
	app()
