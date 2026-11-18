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

from scripts.peer.allocation import AllocationMetadataError, validate_allocation_capacity
from scripts.peer.scheduler import Allocation, NeuSbox, ResourceRequest

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)

RESULT_MARKER = "NEUTRONRT_PEER_RESULT="
DEFAULT_SIZES = (1024, 4096, 65536, 1048576, 16777216)
TORCH_MODES = ("gloo-cpu", "nccl-gpu", "nccl-staged")
RDMA_MODES = ("rdma-host", "rdma-gpu")


def _resource_request(
	mode: str,
	*,
	cpu_cpus: int = 4,
	cpu_memory_gib: int = 8,
	gpu_cpus: int = 4,
	gpu_memory_gib: int = 16,
) -> ResourceRequest:
	if mode in TORCH_MODES + RDMA_MODES:
		gpu = mode in ("nccl-gpu", "nccl-staged", "rdma-gpu")
	else:
		raise ValueError(f"Unsupported peer mode: {mode}")
	return ResourceRequest(
		cpu=gpu_cpus if gpu else cpu_cpus,
		memory_gib=gpu_memory_gib if gpu else cpu_memory_gib,
		gpus=1 if gpu else 0,
	)


def _worker_execution_command(command: list[str], *, gpu: bool, python: str) -> list[str]:
	if gpu:
		return [python, "-m", "scripts.peer.gpu_exec", "--", *command]
	return command


def _scheduler_worker_command(command: list[str], *, gpu: bool, python: str) -> str:
	return "exec " + shlex.join(_worker_execution_command(command, gpu=gpu, python=python))


def _run_local_worker(command: list[str], *, cwd: Path, timeout: float, log_path: Path, label: str) -> str:
	try:
		completed = subprocess.run(
			command,
			cwd=cwd,
			capture_output=True,
			text=True,
			timeout=timeout,
			check=False,
		)
	except subprocess.TimeoutExpired as error:
		stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
		stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
		log_path.write_text(stdout + stderr + f"\n{label} worker timed out after {timeout}s")
		raise TimeoutError(f"{label} worker timed out after {timeout}s; see {log_path}") from error
	log_path.write_text(completed.stdout + completed.stderr)
	if completed.returncode != 0:
		raise RuntimeError(f"{label} worker returned {completed.returncode}; see {log_path}")
	return completed.stdout


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
			return dict(json.loads(line[len(RESULT_MARKER) :]))
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
	*,
	scheduler: NeuSbox | None = None,
	cpu_cpus: int = 4,
	cpu_memory_gib: int = 8,
	gpu_cpus: int = 4,
	gpu_memory_gib: int = 16,
	monitor_window: float = 3600,
	inherited_allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
	if mode in ("nccl-gpu", "nccl-staged") and (local_device != 0 or peer_device != 0):
		raise ValueError("GPU peer workers must use scheduler-visible cuda:0")
	log_dir.mkdir(parents=True, exist_ok=True)
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
	scheduler = scheduler or NeuSbox()
	request = _resource_request(
		mode,
		cpu_cpus=cpu_cpus,
		cpu_memory_gib=cpu_memory_gib,
		gpu_cpus=gpu_cpus,
		gpu_memory_gib=gpu_memory_gib,
	)
	owned: list[Allocation] = []
	peer_task = local_task = None
	local_log_written = False
	try:
		peer_task = scheduler.submit(
			peer_host,
			request,
			_scheduler_worker_command(peer, gpu=request.gpus == 1, python=peer_python),
			cwd=peer_root,
			wall_seconds=timeout,
			ledger=log_dir / "tasks" / f"{mode}.peer.json",
			label=f"{mode}-peer",
		)
		owned.append(peer_task)
		scheduler.wait_running([peer_task], monitor_window)
		if inherited_allocation is None:
			local_task = scheduler.submit(
				None,
				request,
				_scheduler_worker_command(local, gpu=request.gpus == 1, python=local_python),
				cwd=local_root,
				wall_seconds=timeout,
				ledger=log_dir / "tasks" / f"{mode}.local.json",
				label=f"{mode}-local",
			)
			owned.append(local_task)
			scheduler.wait_running(owned, monitor_window)
			scheduler.wait_completed(owned, monitor_window)
			local_stdout, _ = scheduler.result(local_task)
			local_task_id = local_task.task_id
		else:
			local_command = _worker_execution_command(local, gpu=request.gpus == 1, python=local_python)
			local_stdout = _run_local_worker(
				local_command,
				cwd=local_root,
				timeout=timeout,
				log_path=log_dir / f"{mode}.local.log",
				label=mode,
			)
			local_log_written = True
			scheduler.wait_completed([peer_task], monitor_window)
			local_task_id = f"inherited:{inherited_allocation['name']}"
		peer_output, _ = scheduler.result(peer_task)
		if not local_log_written:
			log_dir.joinpath(f"{mode}.local.log").write_text(local_stdout)
		log_dir.joinpath(f"{mode}.peer.log").write_text(peer_output)
		result = _parse_result(local_stdout)
	except Exception as error:
		cleanup_errors = scheduler.cleanup(owned)
		if cleanup_errors:
			error = RuntimeError(f"{error}; cleanup failures: {'; '.join(cleanup_errors)}")
		raise RuntimeError(f"{mode} failed: {error}; see logs in {log_dir}") from error
	finally:
		log_dir.mkdir(parents=True, exist_ok=True)
	if peer_task is None or (local_task is None and inherited_allocation is None):
		raise RuntimeError(f"{mode} did not create both scheduler tasks")
	result["peer_host"] = peer_host
	result["task_ids"] = {"local": local_task_id, "peer": peer_task.task_id}
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
	*,
	peer_addr: str | None = None,
	local_root: Path | None = None,
	peer_root: Path | None = None,
	scheduler: NeuSbox | None = None,
	cpu_cpus: int = 4,
	cpu_memory_gib: int = 8,
	gpu_cpus: int = 4,
	gpu_memory_gib: int = 16,
	monitor_window: float = 3600,
	local_python: str = "python",
	peer_python: str = "python",
	inherited_allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
	if mode not in RDMA_MODES:
		raise ValueError(f"Unsupported RDMA mode: {mode}")
	if mode == "rdma-gpu" and peer_device != 0:
		raise ValueError("GPU peer workers must use scheduler-visible cuda:0")
	if peer_addr is None:
		raise ValueError("RDMA measurements require the peer network address")
	scheduler = scheduler or NeuSbox()
	local_root = Path.cwd() if local_root is None else local_root
	peer_root = Path.cwd() if peer_root is None else peer_root
	request = _resource_request(
		mode,
		cpu_cpus=cpu_cpus,
		cpu_memory_gib=cpu_memory_gib,
		gpu_cpus=gpu_cpus,
		gpu_memory_gib=gpu_memory_gib,
	)
	rows: list[dict[str, Any]] = []
	log_dir.mkdir(parents=True, exist_ok=True)
	for size in sizes:
		for program, metric in (("ib_write_bw", "bandwidth"), ("ib_write_lat", "latency")):
			stamp = uuid4().hex
			remote_json = f"/tmp/neutronrt-peer-{stamp}.json"
			gpu_device = 0 if mode == "rdma-gpu" else None
			server = _perftest_command(program, device, rdma_port, size, iterations, gpu_device, Path(remote_json))
			client_json = log_dir / f"{mode}-{metric}-{size}.json"
			client = _perftest_command(program, device, rdma_port, size, iterations, gpu_device, client_json)
			client.append(peer_addr)
			owned: list[Allocation] = []
			row: dict[str, Any] = {
				"mode": mode,
				"metric": metric,
				"bytes": size,
				"client_json": str(client_json),
				"rdma_port": rdma_port,
				"tunnel_port": tunnel_port,
			}
			try:
				peer_task = scheduler.submit(
					peer_host,
					request,
					_scheduler_worker_command(server, gpu=request.gpus == 1, python=peer_python),
					cwd=peer_root,
					wall_seconds=timeout,
					ledger=log_dir / "tasks" / f"{mode}-{metric}-{size}.peer.json",
					label=f"{mode}-{metric}-{size}-peer",
				)
				owned.append(peer_task)
				scheduler.wait_running([peer_task], monitor_window)
				if inherited_allocation is None:
					local_task = scheduler.submit(
						None,
						request,
						_scheduler_worker_command(client, gpu=request.gpus == 1, python=local_python),
						cwd=local_root,
						wall_seconds=timeout,
						ledger=log_dir / "tasks" / f"{mode}-{metric}-{size}.local.json",
						label=f"{mode}-{metric}-{size}-local",
					)
					owned.append(local_task)
					scheduler.wait_running(owned, monitor_window)
					scheduler.wait_completed(owned, monitor_window)
					client_output, _ = scheduler.result(local_task)
					local_task_id = local_task.task_id
				else:
					client_output = _run_local_worker(
						_worker_execution_command(client, gpu=request.gpus == 1, python=local_python),
						cwd=local_root,
						timeout=timeout,
						log_path=log_dir / f"{mode}-{metric}-{size}.client.log",
						label=f"{mode}-{metric}-{size} client",
					)
					scheduler.wait_completed([peer_task], monitor_window)
					local_task_id = f"inherited:{inherited_allocation['name']}"
				server_output, _ = scheduler.result(peer_task)
				if inherited_allocation is None:
					log_dir.joinpath(f"{mode}-{metric}-{size}.client.log").write_text(client_output)
				log_dir.joinpath(f"{mode}-{metric}-{size}.server.log").write_text(server_output)
				row["task_ids"] = {"local": local_task_id, "peer": peer_task.task_id}
				row["client_output"] = client_output
				row["server_output"] = server_output
			except Exception as error:
				row["error"] = str(error)
				cleanup_errors = scheduler.cleanup(owned)
				if cleanup_errors:
					row["cleanup_error"] = "; ".join(cleanup_errors)
			if client_json.exists():
				row["report"], parse_note = _read_perftest_report(client_json)
				if parse_note is not None:
					row["report_parse_note"] = parse_note
			rows.append(row)
	return {"kind": "perftest", "mode": mode, "device": device, "rows": rows}


def _install_peer_tool(peer_host: str, peer_root: Path, overwrite: bool) -> None:
	files = [
		Path(__file__),
		Path(__file__).with_name("scheduler.py"),
		Path(__file__).with_name("allocation.py"),
		Path(__file__).with_name("gpu_exec.py"),
		Path(__file__).with_name("__init__.py"),
	]
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
					"-o",
					"ServerAliveInterval=5",
					"-o",
					"ServerAliveCountMax=2",
					peer_host,
					"test",
					"-e",
					str(destination),
				],
				check=False,
				timeout=15,
			).returncode
			== 0
		)
		if exists and not overwrite:
			continue
		subprocess.run(
			[
				"scp",
				"-q",
				"-o",
				"BatchMode=yes",
				"-o",
				"ConnectTimeout=5",
				str(source),
				f"{peer_host}:{destination}",
			],
			check=True,
			timeout=15,
		)


@app.command()
def inspect(
	peer_python: Annotated[str | None, typer.Option(help="Explicit frozen peer Python for package metadata")] = None,
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
		"peer_cpu": _remote_output(peer_host, "lscpu -J"),
		"peer_memory": _remote_output(peer_host, "free -b"),
		"peer_gpus": _remote_output(
			peer_host, "nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader"
		),
		"peer_scheduler": _remote_output(peer_host, "command -v neu-sbox || true"),
		"peer_graph_python_candidates": _remote_output(
			peer_host,
			"ls /home/lhr/miniconda/envs/graph/bin/python "
			"/home/hdd/lvhr/miniconda/envs/graph/bin/python 2>/dev/null || true",
		),
	}
	if peer_python:
		code = (
			"import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.executable,"
			"'packages':{n:m.version(n) for n in ['torch','dgl','numpy','grpcio','protobuf']}}))"
		)
		result["peer_environment"] = json.loads(_remote_output(peer_host, shlex.join([peer_python, "-c", code])))
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
	cpu_cpus: Annotated[int, typer.Option("--cpu-cpus", min=1)] = 4,
	cpu_memory_gib: Annotated[int, typer.Option("--cpu-memory-gib", min=1)] = 8,
	gpu_cpus: Annotated[int, typer.Option("--gpu-cpus", min=1)] = 4,
	gpu_memory_gib: Annotated[int, typer.Option("--gpu-memory-gib", min=1)] = 16,
	monitor_window: Annotated[float, typer.Option("--monitor-window", min=1, max=3600)] = 3600,
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
	if any(mode in ("nccl-gpu", "nccl-staged", "rdma-gpu") for mode in chosen_modes) and (
		local_device != 0 or peer_device != 0
	):
		raise typer.BadParameter("GPU modes use scheduler-visible cuda:0; do not pass a host GPU index")
	scheduler = NeuSbox()
	try:
		inherited_allocation, inherited_gpu_uuids = scheduler.current_allocation()
	except AllocationMetadataError as error:
		raise typer.BadParameter(f"Could not validate the current Neu Box allocation: {error}") from error
	required_requests = [
		_resource_request(
			mode,
			cpu_cpus=cpu_cpus,
			cpu_memory_gib=cpu_memory_gib,
			gpu_cpus=gpu_cpus,
			gpu_memory_gib=gpu_memory_gib,
		)
		for mode in chosen_modes
	]
	if inherited_allocation is not None:
		required = ResourceRequest(
			cpu=max(request.cpu for request in required_requests),
			memory_gib=max(request.memory_gib for request in required_requests),
			gpus=max(request.gpus for request in required_requests),
		)
		try:
			validate_allocation_capacity(
				inherited_allocation,
				cpu=required.cpu,
				memory_gib=required.memory_gib,
				gpus=required.gpus,
			)
		except AllocationMetadataError as error:
			raise typer.BadParameter(f"Current allocation cannot run the selected modes: {error}") from error
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
		"local_allocation": (
			{
				"name": inherited_allocation["name"],
				"cpu": inherited_allocation["cpu"],
				"mem": inherited_allocation["mem"],
				"devices": inherited_allocation["devices"],
				"gpu_uuids": inherited_gpu_uuids,
			}
			if inherited_allocation is not None
			else None
		),
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
					scheduler=scheduler,
					cpu_cpus=cpu_cpus,
					cpu_memory_gib=cpu_memory_gib,
					gpu_cpus=gpu_cpus,
					gpu_memory_gib=gpu_memory_gib,
					monitor_window=monitor_window,
					inherited_allocation=inherited_allocation,
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
					peer_addr=peer_addr,
					local_root=local_root,
					peer_root=peer_root,
					scheduler=scheduler,
					cpu_cpus=cpu_cpus,
					cpu_memory_gib=cpu_memory_gib,
					gpu_cpus=gpu_cpus,
					gpu_memory_gib=gpu_memory_gib,
					monitor_window=monitor_window,
					local_python=local_python,
					peer_python=peer_python,
					inherited_allocation=inherited_allocation,
				)
			result["modes"].append(measurement)
		except Exception as error:  # Keep a failed transport visible beside successful modes.
			result["modes"].append({"mode": mode, "error": str(error)})
	result["logs"] = str(log_dir)
	output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
	typer.echo(str(output))


if __name__ == "__main__":
	app()
