"""Serve one graph-storage partition over an authenticated local listener."""

from __future__ import annotations

import json
import os
import socket
import time
from multiprocessing.connection import Listener
from pathlib import Path
from threading import Thread
from typing import Annotated

import numpy as np
import psutil
import typer

from lib.storage.base import BaseTopology
from lib.storage.capacity import GIB, StorageNodeBudget, partition_base_bytes
from lib.storage.remote import base_fingerprint
from lib.storage.shard import TopologyShard


def serve(
	connection,
	*,
	base: BaseTopology,
	start: int,
	stop: int,
	overlay_bytes: int,
	response_bytes: int,
	profiling: bool,
):
	shard = TopologyShard(base, start, stop, max_overlay_bytes=overlay_bytes)
	operations: dict[str, int] = {}
	seconds: dict[str, float] = {}
	shared_bytes = 0
	connection.send((True, "ready"))
	try:
		while True:
			command, *args = connection.recv()
			if command == "close":
				return
			started = time.perf_counter() if profiling else 0.0
			try:
				result: object
				if command in ("rows", "sample_rows"):
					direction, nodes = args[:2]
					fanout, seed = args[2:] if command == "sample_rows" else (-1, 0)
					counts = shard.rows[direction].degrees(nodes)
					selected = np.minimum(counts, fanout) if fanout != -1 else counts
					elements = nodes.size + int(selected.sum())
					if elements * 8 > response_bytes:
						raise MemoryError("Selected adjacency exceeds the shared response budget")
					buffer = np.empty(elements, dtype=np.int64)
					shard.fill(direction, nodes, buffer, fanout=fanout, seed=seed)
					shared_bytes += int(elements * 8)
					result = ("remote-rows", buffer[: nodes.size], buffer[nodes.size :])
				elif command == "degrees":
					result = shard.rows[args[0]].degrees(args[1])
				elif command == "exists":
					result = shard.rows["out"].exists(*args)
				elif command == "multiplicities":
					result = shard.rows["out"].multiplicities(*args)
				elif command == "prepare":
					result = shard.prepare(args[0], args[1], delete=args[2], revision=args[3])
				elif command == "commit":
					shard.commit(args[0])
					result = None
				elif command == "abort":
					shard.abort()
					result = None
				elif command == "stats":
					memory = psutil.Process().memory_full_info()
					result = {
						"revision": shard.revision,
						"operations": dict(operations),
						"seconds": dict(seconds),
						"shared_response_bytes": shared_bytes,
						"cpu_seconds": time.process_time(),
						"rss_bytes": int(memory.rss),
						"pss_bytes": getattr(memory, "pss", None),
						"uss_bytes": getattr(memory, "uss", None),
						"overlay_bytes": sum(rows.bytes_used for rows in shard.rows.values()),
						"overlay_metadata_bytes": sum(rows.overlay_metadata_bytes for rows in shard.rows.values()),
						"overlay_rows": sum(len(rows.overlays) for rows in shard.rows.values()),
						"response_buffers": 0,
						"torch_loaded": False,
						"cuda_initialized": False,
					}
				elif command == "rss":
					result = {"rss_bytes": int(psutil.Process().memory_info().rss)}
				else:
					raise ValueError("Unknown remote storage operation")
				if profiling and command not in ("stats", "rss"):
					operations[command] = operations.get(command, 0) + 1
					seconds[command] = seconds.get(command, 0.0) + time.perf_counter() - started
				connection.send((True, result))
			except Exception as error:
				connection.send((False, (type(error).__name__, str(error))))
	finally:
		connection.close()


def validate_init(
	args: list,
	*,
	base: BaseTopology,
	start: int,
	stop: int,
	capacity_bytes: int,
	reserve_bytes: int,
	overlay_bytes: int,
	response_bytes: int,
) -> None:
	if len(args) != 10 or tuple(args[1:5]) != (
		base.nodes,
		base.edges,
		start,
		stop,
	):
		raise ValueError("Remote storage init does not match this topology partition")
	if args[9] != base_fingerprint(base):
		raise ValueError("Remote storage init does not match the exported graph content")
	if tuple(args[5:9]) != (overlay_bytes, response_bytes, capacity_bytes, reserve_bytes):
		raise ValueError("Remote storage init does not match this node budget")
	StorageNodeBudget("remote", capacity_bytes, reserve_bytes)
	if partition_base_bytes(base, start, stop) > capacity_bytes - reserve_bytes:
		raise MemoryError("Mapped adjacency exceeds the storage node usable capacity")


app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.command()
def main(
	base: Annotated[Path, typer.Option("--base")],
	nodes: Annotated[int, typer.Option("--nodes")],
	edges: Annotated[int, typer.Option("--edges")],
	start: Annotated[int, typer.Option("--start")],
	stop: Annotated[int, typer.Option("--stop")],
	host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
	port: Annotated[int, typer.Option("--port")] = 27182,
	authkey: Annotated[str, typer.Option("--authkey")] = "neutronrt",
	overlay_bytes: Annotated[int, typer.Option("--overlay-bytes", min=0)] = 256 * 1024**2,
	response_bytes: Annotated[int, typer.Option("--response-bytes", min=0)] = 256 * 1024**2,
	capacity_bytes: Annotated[int, typer.Option("--capacity-bytes", min=1)] = 512 * GIB,
	reserve_bytes: Annotated[int, typer.Option("--reserve-bytes", min=0)] = 64 * GIB,
	profiling: Annotated[bool, typer.Option("--profiling/--no-profiling")] = True,
):
	if not 0 <= start < stop <= nodes or min(nodes, edges) < 0:
		raise typer.BadParameter("Invalid node range or graph dimensions")
	try:
		StorageNodeBudget("remote", capacity_bytes, reserve_bytes)
	except ValueError as error:
		raise typer.BadParameter(str(error)) from error
	base_topology = BaseTopology(base.resolve(), nodes, edges)
	try:
		if partition_base_bytes(base_topology, start, stop) > capacity_bytes - reserve_bytes:
			raise typer.BadParameter("Mapped adjacency exceeds the storage node usable capacity")
	except (FileNotFoundError, ValueError) as error:
		raise typer.BadParameter(str(error)) from error
	listener = Listener((host, port), authkey=authkey.encode())
	print(json.dumps({"host": socket.gethostname(), "listen": listener.address, "pid": os.getpid()}), flush=True)

	def handle(connection):
		try:
			command, *args = connection.recv()
			if command != "ping":
				raise ValueError("First storage command must be ping")
			connection.send((True, "ready"))
			command, *args = connection.recv()
			if command != "init":
				raise ValueError("Second storage command must be init")
			validate_init(
				args,
				base=base_topology,
				start=start,
				stop=stop,
				capacity_bytes=capacity_bytes,
				reserve_bytes=reserve_bytes,
				overlay_bytes=overlay_bytes,
				response_bytes=response_bytes,
			)
			serve(
				connection,
				base=base_topology,
				start=start,
				stop=stop,
				overlay_bytes=overlay_bytes,
				response_bytes=response_bytes,
				profiling=profiling,
			)
		except Exception as error:
			connection.send((False, (type(error).__name__, str(error))))
			connection.close()

	try:
		while True:
			connection = listener.accept()
			Thread(target=handle, args=(connection,), daemon=True).start()
	finally:
		listener.close()


if __name__ == "__main__":
	app()
