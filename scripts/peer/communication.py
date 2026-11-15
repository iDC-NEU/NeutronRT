"""Measure Gloo, NCCL, and InfiniBand communication with a peer node.

The ``run`` command is the coordinator. It starts a matching ``worker`` on the
peer checkout over SSH and keeps generated logs/results outside the source tree.
``rdma-host`` uses CPU memory, ``rdma-gpu`` uses GPUDirect RDMA buffers, and
``nccl-staged`` includes explicit host-to-device/device-to-host staging around
the NCCL transfer.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from statistics import median
from typing import Annotated, Any
from uuid import uuid4

import torch
import typer
from torch import distributed as dist

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)

RESULT_MARKER = "NEUTRONRT_PEER_RESULT="
DEFAULT_SIZES = (1024, 4096, 65536, 1048576, 16777216)
TORCH_MODES = ("gloo-cpu", "nccl-gpu", "nccl-staged")
RDMA_MODES = ("rdma-host", "rdma-gpu")


def _quantile(values: list[float], fraction: float) -> float:
	if not values:
		raise ValueError("Cannot summarize an empty sample")
	ordered = sorted(values)
	position = (len(ordered) - 1) * fraction
	low = int(position)
	high = min(low + 1, len(ordered) - 1)
	weight = position - low
	return ordered[low] * (1 - weight) + ordered[high] * weight


def _interface_address(interface: str) -> str:
	completed = subprocess.run(
		["ip", "-4", "-o", "addr", "show", "dev", interface],
		check=True,
		capture_output=True,
		text=True,
	)
	match = re.search(r"inet\s+(\d+(?:\.\d+){3})/", completed.stdout)
	if match is None:
		raise RuntimeError(f"No IPv4 address found on interface {interface}")
	return match.group(1)


def _remote_output(host: str, command: str, timeout: float = 15) -> str:
	completed = subprocess.run(
		["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command],
		check=True,
		capture_output=True,
		text=True,
		timeout=timeout,
	)
	return completed.stdout.strip()


def _remote_command(host: str, command: str, timeout: float = 15) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, command],
		capture_output=True,
		text=True,
		timeout=timeout,
		check=False,
	)


def _local_listener(port: int) -> bool:
	completed = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, check=False)
	return re.search(rf":{port}\b", completed.stdout) is not None


def _read_perftest_report(path: Path) -> tuple[dict[str, Any], str | None]:
	text = path.read_text()
	try:
		return json.loads(text), None
	except json.JSONDecodeError:
		# Some perftest releases leave a trailing comma in nested result objects.
		cleaned = re.sub(r",\s*([}\]])", r"\1", text)
		return json.loads(cleaned), "normalized trailing commas emitted by perftest"


def _remote_interface_address(host: str, interface: str) -> str:
	return _remote_output(
		host,
		f"ip -4 -o addr show dev {shlex.quote(interface)} | awk '{{print $4}}' | cut -d/ -f1",
	).splitlines()[0]


def _git_info(root: Path, host: str | None = None) -> dict[str, Any]:
	command = "git -C %s rev-parse HEAD; git -C %s status --porcelain=v1 --branch" % (
		shlex.quote(str(root)),
		shlex.quote(str(root)),
	)
	if host is None:
		completed = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
		text = completed.stdout
	else:
		text = _remote_output(host, command)
	lines = text.splitlines()
	return {"commit": lines[0] if lines else None, "status": lines[1:], "clean": len(lines) <= 1}


def _set_transport_environment(mode: str, interface: str | None) -> None:
	if interface:
		os.environ["GLOO_SOCKET_IFNAME"] = interface
		os.environ["NCCL_SOCKET_IFNAME"] = interface
	if mode.startswith("nccl"):
		os.environ.setdefault("NCCL_IB_DISABLE", "0")
		os.environ.setdefault("NCCL_IB_HCA", "mlx5_0:1")
		os.environ.setdefault("NCCL_DEBUG", "WARN")


def _sync(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _torch_exchange(
	mode: str,
	rank: int,
	device: torch.device,
	send: torch.Tensor,
	recv: torch.Tensor,
	send_cpu: torch.Tensor | None,
	recv_cpu: torch.Tensor | None,
) -> None:
	if mode == "nccl-staged":
		assert send_cpu is not None and recv_cpu is not None
		if rank == 0:
			send.copy_(send_cpu, non_blocking=True)
			_sync(device)
			dist.send(send, dst=1)
			dist.recv(recv, src=1)
			recv_cpu.copy_(recv, non_blocking=True)
			_sync(device)
		else:
			dist.recv(recv, src=0)
			dist.send(recv, dst=0)
			recv_cpu.copy_(recv, non_blocking=True)
			_sync(device)
		return
	if rank == 0:
		dist.send(send, dst=1)
		dist.recv(recv, src=1)
	else:
		dist.recv(recv, src=0)
		dist.send(recv, dst=0)
	_sync(device)


def _torch_measure(
	mode: str,
	rank: int,
	device_index: int,
	sizes: tuple[int, ...],
	warmup: int,
	iterations: int,
) -> dict[str, Any] | None:
	backend = "gloo" if mode == "gloo-cpu" else "nccl"
	device = torch.device("cpu" if backend == "gloo" else f"cuda:{device_index}")
	if device.type == "cuda":
		torch.cuda.set_device(device)
	rows = []
	for size in sizes:
		send = torch.ones(size, dtype=torch.uint8, device=device)
		recv = torch.empty_like(send)
		send_cpu = recv_cpu = None
		if mode == "nccl-staged":
			send_cpu = torch.ones(size, dtype=torch.uint8, pin_memory=True)
			recv_cpu = torch.empty_like(send_cpu)
		for _ in range(warmup):
			_torch_exchange(mode, rank, device, send, recv, send_cpu, recv_cpu)
		dist.barrier()
		samples = []
		for _ in range(iterations):
			dist.barrier()
			_sync(device)
			started = time.perf_counter()
			_torch_exchange(mode, rank, device, send, recv, send_cpu, recv_cpu)
			_sync(device)
			if rank == 0:
				samples.append(time.perf_counter() - started)
		dist.barrier()
		if rank == 0:
			rtt = median(samples)
			rows.append(
				{
					"bytes": size,
					"iterations": iterations,
					"rtt_p50_seconds": rtt,
					"rtt_p95_seconds": _quantile(samples, 0.95),
					"one_way_latency_p50_us": rtt * 500_000,
					"aggregate_bidirectional_gbps_p50": (2 * size * 8) / rtt / 1e9,
				}
			)
		del send, recv, send_cpu, recv_cpu
		if device.type == "cuda":
			torch.cuda.empty_cache()
	return (
		{
			"kind": "torch-ping-pong",
			"mode": mode,
			"backend": backend,
			"device": str(device),
			"sizes": list(sizes),
			"warmup": warmup,
			"iterations": iterations,
			"rows": rows,
		}
		if rank == 0
		else None
	)


@app.command()
def worker(
	mode: Annotated[str, typer.Option(help="gloo-cpu, nccl-gpu, or nccl-staged")],
	rank: Annotated[int, typer.Option(min=0)],
	master_addr: Annotated[str, typer.Option()],
	master_port: Annotated[int, typer.Option(min=1, max=65535)],
	world_size: Annotated[int, typer.Option(min=2)] = 2,
	interface: Annotated[str | None, typer.Option()] = None,
	device: Annotated[int, typer.Option(min=0)] = 0,
	sizes: Annotated[list[int] | None, typer.Option("--size", min=1)] = None,
	warmup: Annotated[int, typer.Option(min=0)] = 10,
	iterations: Annotated[int, typer.Option(min=1)] = 50,
	timeout: Annotated[float, typer.Option(min=1)] = 120,
):
	"""Internal rank worker used by ``run``."""
	if mode not in TORCH_MODES:
		raise typer.BadParameter(f"Unsupported torch mode: {mode}")
	if rank >= world_size:
		raise typer.BadParameter("rank must be smaller than world_size")
	chosen_sizes = tuple(sizes or DEFAULT_SIZES)
	_set_transport_environment(mode, interface)
	backend = "gloo" if mode == "gloo-cpu" else "nccl"
	dist.init_process_group(
		backend=backend,
		init_method=f"tcp://{master_addr}:{master_port}",
		rank=rank,
		world_size=world_size,
		timeout=timedelta(seconds=timeout),
	)
	try:
		result = _torch_measure(mode, rank, device, chosen_sizes, warmup, iterations)
		if result is not None:
			print(RESULT_MARKER + json.dumps(result, separators=(",", ":")), flush=True)
		dist.barrier()
	finally:
		dist.destroy_process_group()


def _parse_result(text: str) -> dict[str, Any]:
	for line in reversed(text.splitlines()):
		if line.startswith(RESULT_MARKER):
			return json.loads(line[len(RESULT_MARKER) :])
	raise RuntimeError(f"No {RESULT_MARKER} marker in worker output")


def _worker_command(
	python: str,
	mode: str,
	rank: int,
	master_addr: str,
	master_port: int,
	interface: str,
	device: int,
	sizes: tuple[int, ...],
	warmup: int,
	iterations: int,
	timeout: float,
) -> list[str]:
	command = [
		python,
		"-m",
		"scripts.peer.communication",
		"worker",
		"--mode",
		mode,
		"--rank",
		str(rank),
		"--master-addr",
		master_addr,
		"--master-port",
		str(master_port),
		"--interface",
		interface,
		"--device",
		str(device),
		"--warmup",
		str(warmup),
		"--iterations",
		str(iterations),
		"--timeout",
		str(timeout),
	]
	for size in sizes:
		command.extend(["--size", str(size)])
	return command


def _run_torch_pair(
	local_root: Path,
	local_python: str,
	peer_host: str,
	peer_root: Path,
	peer_python: str,
	mode: str,
	master_addr: str,
	master_port: int,
	local_interface: str,
	peer_interface: str,
	local_device: int,
	peer_device: int,
	sizes: tuple[int, ...],
	warmup: int,
	iterations: int,
	timeout: float,
	log_dir: Path,
) -> dict[str, Any]:
	local = _worker_command(
		local_python,
		mode,
		0,
		master_addr,
		master_port,
		local_interface,
		local_device,
		sizes,
		warmup,
		iterations,
		timeout,
	)
	peer = _worker_command(
		peer_python,
		mode,
		1,
		master_addr,
		master_port,
		peer_interface,
		peer_device,
		sizes,
		warmup,
		iterations,
		timeout,
	)
	remote_command = "cd %s && %s" % (shlex.quote(str(peer_root)), shlex.join(peer))
	peer_process = subprocess.Popen(
		["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", peer_host, remote_command],
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
	)
	local_process = None
	peer_output = ""
	peer_returncode = None
	local_error = None
	try:
		try:
			local_process = subprocess.run(
				local,
				cwd=local_root,
				capture_output=True,
				text=True,
				timeout=timeout,
				check=False,
			)
		except subprocess.TimeoutExpired as error:
			local_error = f"local worker timed out after {timeout}s"
			if error.stdout:
				local_error += f": {error.stdout}"
		try:
			peer_output, peer_returncode = peer_process.communicate(timeout=timeout)
			peer_returncode = peer_process.returncode
		except subprocess.TimeoutExpired:
			peer_process.kill()
			peer_output, _ = peer_process.communicate()
			peer_returncode = peer_process.returncode
	finally:
		if peer_process.poll() is None:
			peer_process.kill()
			peer_output, _ = peer_process.communicate()
			peer_returncode = peer_process.returncode
	local_stdout = "" if local_process is None else local_process.stdout
	local_stderr = "" if local_process is None else local_process.stderr
	log_dir.joinpath(f"{mode}.local.log").write_text(local_stdout + local_stderr + (local_error or ""))
	log_dir.joinpath(f"{mode}.peer.log").write_text(peer_output)
	if local_error or local_process is None or local_process.returncode != 0 or peer_returncode != 0:
		raise RuntimeError(
			f"{mode} failed: {local_error or f'local={local_process.returncode}, peer={peer_returncode}'}; "
			f"see logs in {log_dir}"
		)
	result = _parse_result(local_stdout)
	result["peer_host"] = peer_host
	return result


def _perftest_command(
	program: str,
	device: str,
	port: int,
	size: int,
	iterations: int,
	gpu: int | None,
	output: Path,
) -> list[str]:
	command = [
		program,
		"-d",
		device,
		"-i",
		"1",
		"-p",
		str(port),
		"-s",
		str(size),
		"-n",
		str(iterations),
		"--out_json",
		"--out_json_file",
		str(output),
	]
	if program == "ib_write_bw":
		command.extend(["--output", "bandwidth", "--report_gbits"])
	if gpu is not None:
		command.extend([f"--use_cuda={gpu}", "--use_cuda_dmabuf"])
	return command


def _run_rdma(
	peer_host: str,
	mode: str,
	device: str,
	peer_device: int,
	sizes: tuple[int, ...],
	iterations: int,
	timeout: float,
	log_dir: Path,
	rdma_port: int,
	tunnel_port: int,
) -> dict[str, Any]:
	gpu = peer_device if mode == "rdma-gpu" else None
	rows = []
	for size in sizes:
		for program, metric in (("ib_write_bw", "bandwidth"), ("ib_write_lat", "latency")):
			stamp = uuid4().hex
			remote_json = f"/tmp/neutronrt-peer-{stamp}.json"
			remote_pidfile = f"/tmp/neutronrt-peer-{stamp}.pid"
			server = _perftest_command(program, device, rdma_port, size, iterations, gpu, Path(remote_json))
			remote_server = "rm -f %s; echo $$ > %s; exec %s" % (
				shlex.quote(remote_pidfile),
				shlex.quote(remote_pidfile),
				shlex.join(server),
			)
			server_process = subprocess.Popen(
				[
					"ssh",
					"-o",
					"BatchMode=yes",
					"-o",
					"ConnectTimeout=5",
					peer_host,
					remote_server,
				],
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
			)
			client_json = log_dir / f"{mode}-{metric}-{size}.json"
			client = _perftest_command(program, device, tunnel_port, size, iterations, gpu, client_json)
			tunnel_process = None
			completed = None
			server_output = ""
			tunnel_output = ""
			server_returncode = None
			client_error = None
			try:
				listener_deadline = time.monotonic() + min(timeout, 15)
				while time.monotonic() < listener_deadline:
					if server_process.poll() is not None:
						break
					listener = _remote_command(
						peer_host,
						f"ss -ltn '( sport = :{rdma_port} )' | tail -n +2",
						timeout=3,
					)
					if listener.returncode == 0 and listener.stdout.strip():
						break
					time.sleep(0.25)
				else:
					client_error = f"remote perftest did not listen on port {rdma_port}"
				if client_error is None and server_process.poll() is not None:
					client_error = "remote perftest exited before the client started"
				if client_error is None:
					tunnel_process = subprocess.Popen(
						[
							"ssh",
							"-N",
							"-o",
							"BatchMode=yes",
							"-o",
							"ConnectTimeout=5",
							"-o",
							"ExitOnForwardFailure=yes",
							"-L",
							f"{tunnel_port}:127.0.0.1:{rdma_port}",
							peer_host,
						],
						stdout=subprocess.PIPE,
						stderr=subprocess.STDOUT,
						text=True,
					)
					local_deadline = time.monotonic() + min(timeout, 15)
					while time.monotonic() < local_deadline:
						if tunnel_process.poll() is not None:
							break
						if _local_listener(tunnel_port):
							break
						time.sleep(0.1)
					if tunnel_process.poll() is not None:
						tunnel_output, _ = tunnel_process.communicate()
						client_error = "SSH tunnel exited before the perftest client started"
					elif not _local_listener(tunnel_port):
						client_error = f"SSH tunnel did not listen on local port {tunnel_port}"
				if client_error is not None:
					completed = subprocess.CompletedProcess(client, 1, "", client_error)
				try:
					if client_error is None:
						completed = subprocess.run(
							client + ["127.0.0.1"],
							capture_output=True,
							text=True,
							timeout=timeout,
							check=False,
						)
				except subprocess.TimeoutExpired:
					client_error = f"client timed out after {timeout}s"
			finally:
				if tunnel_process is not None and tunnel_process.poll() is None:
					tunnel_process.terminate()
					try:
						tunnel_output, _ = tunnel_process.communicate(timeout=5)
					except subprocess.TimeoutExpired:
						tunnel_process.kill()
						tunnel_output, _ = tunnel_process.communicate()
				elif tunnel_process is not None and not tunnel_output:
					tunnel_output, _ = tunnel_process.communicate()
				cleanup = _remote_command(
					peer_host,
					"if test -s %s; then kill $(cat %s) 2>/dev/null || true; fi; rm -f %s"
					% (
						shlex.quote(remote_pidfile),
						shlex.quote(remote_pidfile),
						shlex.quote(remote_pidfile),
					),
					timeout=10,
				)
				if cleanup.returncode != 0:
					server_output += "\nremote cleanup failed: " + cleanup.stderr
				if server_process.poll() is None:
					server_process.terminate()
					try:
						server_output, _ = server_process.communicate(timeout=5)
					except subprocess.TimeoutExpired:
						server_process.kill()
						server_output, _ = server_process.communicate()
				else:
					server_output, _ = server_process.communicate()
				server_returncode = server_process.returncode
			client_stdout = "" if completed is None else completed.stdout
			client_stderr = "" if completed is None else completed.stderr
			log_dir.joinpath(f"{mode}-{metric}-{size}.client.log").write_text(
				client_stdout + client_stderr + (client_error or "")
			)
			log_dir.joinpath(f"{mode}-{metric}-{size}.server.log").write_text(server_output)
			log_dir.joinpath(f"{mode}-{metric}-{size}.tunnel.log").write_text(tunnel_output)
			result: dict[str, Any] = {
				"mode": mode,
				"metric": metric,
				"bytes": size,
				"client_returncode": None if completed is None else completed.returncode,
				"server_returncode": server_returncode,
				"client_json": str(client_json),
				"rdma_port": rdma_port,
				"tunnel_port": tunnel_port,
			}
			if client_json.exists():
				result["report"], parse_note = _read_perftest_report(client_json)
				if parse_note is not None:
					result["report_parse_note"] = parse_note
			server_failed = server_returncode is not None and server_returncode not in (0, -15, -9)
			if client_error or completed is None or completed.returncode != 0 or server_failed:
				result["error"] = "perftest failed; inspect client/server logs"
			rows.append(result)
	return {"kind": "perftest", "mode": mode, "device": device, "rows": rows}


def _install_peer_tool(peer_host: str, peer_root: Path, overwrite: bool) -> None:
	files = [Path(__file__), Path(__file__).with_name("__init__.py")]
	_remote_output(peer_host, f"mkdir -p {shlex.quote(str(peer_root / 'scripts' / 'peer'))}")
	for source in files:
		destination = peer_root / "scripts" / "peer" / source.name
		exists = (
			subprocess.run(
				[
					"ssh",
					"-o",
					"BatchMode=yes",
					"-o",
					"ConnectTimeout=5",
					peer_host,
					"test",
					"-e",
					str(destination),
				],
				check=False,
			).returncode
			== 0
		)
		if exists and not overwrite:
			continue
		subprocess.run(["scp", "-q", str(source), f"{peer_host}:{destination}"], check=True)


@app.command()
def inspect(
	peer_host: Annotated[str, typer.Option(help="SSH host for the second node")] = "gpunode2",
	peer_root: Annotated[Path, typer.Option(help="Peer checkout containing NeutronRT")] = Path("/home/lhr/NeutronRT"),
):
	"""Check the peer checkout and communication prerequisites."""
	result = {
		"local": _git_info(Path.cwd()),
		"peer": _git_info(peer_root, peer_host),
		"peer_host": peer_host,
		"peer_root": str(peer_root),
		"peer_python": _remote_output(peer_host, "command -v python || command -v python3 || true"),
		"peer_rdma_devices": _remote_output(peer_host, "ibv_devinfo -l 2>/dev/null || true"),
	}
	typer.echo(json.dumps(result, indent=2))


@app.command()
def run(
	output: Annotated[Path, typer.Option(help="New JSON result file; refuses to overwrite")],
	peer_host: Annotated[str, typer.Option()] = "gpunode2",
	peer_root: Annotated[Path, typer.Option()] = Path("/home/lhr/NeutronRT"),
	local_python: Annotated[str, typer.Option()] = sys.executable,
	peer_python: Annotated[str, typer.Option(help="Python executable in the peer graph environment")] = "python",
	interface: Annotated[str, typer.Option(help="Local and peer network interface")] = "ibs10",
	peer_interface: Annotated[str | None, typer.Option()] = None,
	peer_addr: Annotated[str | None, typer.Option(help="Peer IPv4 address on the RDMA network")] = None,
	local_device: Annotated[int, typer.Option(min=0)] = 0,
	peer_device: Annotated[int, typer.Option(min=0)] = 0,
	rdma_device: Annotated[str, typer.Option()] = "mlx5_0",
	modes: Annotated[list[str] | None, typer.Option("--mode")] = None,
	sizes: Annotated[list[int] | None, typer.Option("--size", min=1)] = None,
	warmup: Annotated[int, typer.Option(min=0)] = 10,
	iterations: Annotated[int, typer.Option(min=1)] = 50,
	timeout: Annotated[float, typer.Option(min=1)] = 180,
	master_port: Annotated[int, typer.Option(min=1, max=65535)] = 29600,
	rdma_port: Annotated[int, typer.Option(min=1, max=65535)] = 38515,
	rdma_tunnel_port: Annotated[int, typer.Option(min=1, max=65535)] = 39515,
	install_peer_tool: Annotated[bool, typer.Option(help="Install this tool if absent in the peer checkout")] = True,
	overwrite_peer_tool: Annotated[bool, typer.Option()] = False,
):
	"""Run matched peer communication measurements and write JSON."""
	if output.exists():
		raise typer.BadParameter("Output exists; choose a new path to preserve prior measurements")
	chosen_modes = tuple(modes or ("gloo-cpu", "rdma-host", "rdma-gpu", "nccl-gpu", "nccl-staged"))
	unknown = set(chosen_modes) - set(TORCH_MODES) - set(RDMA_MODES)
	if unknown:
		raise typer.BadParameter(f"Unsupported modes: {sorted(unknown)}")
	chosen_sizes = tuple(sizes or DEFAULT_SIZES)
	peer_interface = peer_interface or interface
	local_addr = _interface_address(interface)
	peer_addr = peer_addr or _remote_interface_address(peer_host, peer_interface)
	local_root = Path.cwd()
	peer_info = _git_info(peer_root, peer_host)
	if install_peer_tool:
		_install_peer_tool(peer_host, peer_root, overwrite_peer_tool)
	output.parent.mkdir(parents=True, exist_ok=True)
	log_dir = output.with_name(output.stem + "-logs")
	log_dir.mkdir(parents=True, exist_ok=False)
	local_info = _git_info(local_root)
	result: dict[str, Any] = {
		"peer_host": peer_host,
		"peer_root": str(peer_root),
		"local_interface": interface,
		"peer_interface": peer_interface,
		"local_address": local_addr,
		"peer_address": peer_addr,
		"local_git": local_info,
		"peer_git": peer_info,
		"same_commit": local_info["commit"] == peer_info["commit"],
		"modes": [],
		"sizes": list(chosen_sizes),
		"warning": "Communication results are independent of NeutronRT algorithm results.",
	}
	for index, mode in enumerate(chosen_modes):
		try:
			if mode in TORCH_MODES:
				measurement = _run_torch_pair(
					local_root,
					local_python,
					peer_host,
					peer_root,
					peer_python,
					mode,
					local_addr,
					master_port + index,
					interface,
					peer_interface,
					local_device,
					peer_device,
					chosen_sizes,
					warmup,
					iterations,
					timeout,
					log_dir,
				)
			else:
				measurement = _run_rdma(
					peer_host,
					mode,
					rdma_device,
					peer_device,
					chosen_sizes,
					iterations,
					timeout,
					log_dir,
					rdma_port + index,
					rdma_tunnel_port + index,
				)
			result["modes"].append(measurement)
		except Exception as error:  # Keep a failed transport visible beside successful modes.
			result["modes"].append({"mode": mode, "error": str(error)})
	result["logs"] = str(log_dir)
	output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
	typer.echo(str(output))


if __name__ == "__main__":
	app()
