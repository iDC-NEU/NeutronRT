"""One reviewable host batch for sockets, workers, CUDA, recovery, RDMA, and timings.

Without --execute, print the plan without probing devices or opening sockets.
The batch installs nothing and never kills pre-existing processes.
"""

import csv
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Any, Dict, Optional, Sequence, Tuple

import psutil
import typer


@dataclass(frozen=True)
class Stage:
	name: str
	command: Tuple[str, ...]
	timeout: int
	dependencies: Tuple[str, ...] = ()
	gpu: bool = False
	peers: bool = False
	rdma: bool = False
	minimum_host_bytes: int = 0
	environment: Tuple[Tuple[str, str], ...] = ()


def stages(data_root: Path, output: Path, *, products: bool = False) -> Tuple[Stage, ...]:
	python = sys.executable
	raw = str(data_root / "arxiv" / "raw")
	checkpoint = str(data_root / "checkpoints" / "sage-arxiv.pt")
	base = ("--data", raw, "--checkpoint", checkpoint)
	plan = [
		Stage("tests", (python, "-m", "pytest", "tests/", "-q"), 1200),
		Stage(
			"arxiv-storage",
			(python, "-m", "scripts.validate.storage", *base, "--output", str(output / "arxiv-storage.json")),
			1200,
			dependencies=("tests",),
		),
		Stage(
			"cpu-accuracy",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/accuracy-cpu.json",
				"--updates",
				"4",
				"--warmup",
				"3",
				"--queries",
				"48603",
				"--query-selection",
				"test",
				"--repetitions",
				"1",
				"--output",
				str(output / "cpu-accuracy"),
			),
			1200,
			dependencies=("tests", "arxiv-storage"),
		),
		Stage(
			"arxiv-retained",
			(
				python,
				"-m",
				"scripts.validate.storage",
				*base,
				"--models",
				"sage",
				"--retain-aggregates",
				"--output",
				str(output / "arxiv-retained.json"),
			),
			1200,
			dependencies=("tests",),
		),
		Stage(
			"cpu-retained-profiles",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/retained-cpu.json",
				"--updates",
				"30",
				"--changes",
				"1000",
				"--pool-size",
				"2048",
				"--repetitions",
				"5",
				"--output",
				str(output / "cpu-retained-profiles"),
			),
			3600,
			dependencies=("tests", "arxiv-retained"),
		),
		Stage(
			"arxiv-aggregate-only",
			(
				python,
				"-m",
				"scripts.validate.storage",
				*base,
				"--models",
				"sage",
				"--retain-aggregates",
				"--aggregate-only",
				"--worker-bytes",
				"1048576",
				"--batch-size",
				"1024",
				"--output",
				str(output / "arxiv-aggregate-only.json"),
			),
			1200,
			dependencies=("tests",),
		),
		Stage(
			"cpu-aggregate-only-profiles",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/aggregate-only-cpu.json",
				"--updates",
				"30",
				"--changes",
				"1000",
				"--pool-size",
				"2048",
				"--queries",
				"1024",
				"--worker-bytes",
				"33554432",
				"--storage-bytes",
				"268435456",
				"--repetitions",
				"5",
				"--output",
				str(output / "cpu-aggregate-only-profiles"),
			),
			5400,
			dependencies=("tests", "arxiv-aggregate-only"),
		),
		Stage(
			"cpu-aggregate-only-accuracy",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/aggregate-only-accuracy-cpu.json",
				"--updates",
				"4",
				"--warmup",
				"3",
				"--queries",
				"48603",
				"--query-selection",
				"test",
				"--worker-bytes",
				"33554432",
				"--storage-bytes",
				"268435456",
				"--repetitions",
				"1",
				"--output",
				str(output / "cpu-aggregate-only-accuracy"),
			),
			1800,
			dependencies=("tests", "arxiv-aggregate-only"),
		),
		Stage(
			"cpu-storage-profiles",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/storage-cpu.json",
				"--updates",
				"30",
				"--changes",
				"1000",
				"--pool-size",
				"2048",
				"--repetitions",
				"5",
				"--output",
				str(output / "cpu-storage-profiles"),
			),
			5400,
			dependencies=("tests", "arxiv-storage"),
		),
		Stage(
			"arxiv-rpc",
			(
				python,
				"-m",
				"scripts.validate.inference_rpc",
				*base,
				"--device",
				"cuda:0",
				"--models",
				"sage",
				"--models",
				"gat",
				"--models",
				"gcn",
				"--output",
				str(output / "arxiv-rpc.json"),
			),
			900,
			gpu=True,
		),
		Stage(
			"arxiv-recovery",
			(
				python,
				"-m",
				"scripts.validate.worker_recovery",
				*base,
				"--device",
				"cuda:0",
				"--models",
				"sage",
				"--models",
				"gat",
				"--models",
				"gcn",
				"--output",
				str(output / "arxiv-recovery.json"),
			),
			1200,
			dependencies=("arxiv-rpc",),
			gpu=True,
		),
		Stage(
			"profiles",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/default.json",
				"--output",
				str(output / "profiles"),
				"--repetitions",
				"5",
				"--trace",
				"--trace-tasks",
				"20",
			),
			5400,
			dependencies=("tests", "arxiv-rpc"),
			gpu=True,
		),
		Stage(
			"larger-batches",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/large.json",
				"--changes",
				"1000",
				"--pool-size",
				"2048",
				"--updates",
				"9",
				"--repetitions",
				"5",
				"--output",
				str(output / "larger-batches"),
			),
			5400,
			dependencies=("tests", "arxiv-rpc"),
			gpu=True,
		),
		Stage(
			"test-split-accuracy",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/accuracy.json",
				"--updates",
				"4",
				"--warmup",
				"3",
				"--queries",
				"48603",
				"--query-selection",
				"test",
				"--repetitions",
				"1",
				"--output",
				str(output / "test-split-accuracy"),
			),
			1200,
			dependencies=("tests", "arxiv-rpc"),
			gpu=True,
		),
		Stage(
			"peer-profiles",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/peer.json",
				"--output",
				str(output / "peer-profiles"),
				"--repetitions",
				"5",
			),
			2400,
			dependencies=("tests", "arxiv-rpc"),
			gpu=True,
			peers=True,
		),
		Stage(
			"concurrent-sustained",
			(
				python,
				"-m",
				"scripts.evaluate.profile",
				*base,
				"--cases",
				"examples/profiles/sustained.json",
				"--updates",
				"120",
				"--changes",
				"1000",
				"--pool-size",
				"2048",
				"--repetitions",
				"5",
				"--output",
				str(output / "concurrent-sustained"),
			),
			5400,
			dependencies=("tests", "arxiv-rpc"),
			gpu=True,
			peers=True,
		),
		Stage(
			"forced-rdma",
			(python, "-m", "pytest", "tests/dist/test_transport.py::test_cpu_gpu_and_gpu_peer_transport", "-q"),
			240,
			dependencies=("tests",),
			gpu=True,
			peers=True,
			rdma=True,
			environment=(
				("NCCL_NET", "IB"),
				("NCCL_IB_HCA", "mlx5_0"),
				("NCCL_NET_GDR_LEVEL", "0"),
				("NCCL_P2P_DISABLE", "1"),
				("NCCL_SHM_DISABLE", "1"),
			),
		),
	]
	plan.extend(
		[
			Stage(
				"arxiv-delta",
				(
					python,
					"-m",
					"scripts.validate.storage",
					*base,
					"--models",
					"sage",
					"--retain-aggregates",
					"--incremental-aggregates",
					"--delta-max-steps",
					"2",
					"--batch-size",
					"2048",
					"--worker-bytes",
					str(32 * 1024**2),
					"--output",
					str(output / "arxiv-delta.json"),
				),
				1200,
				dependencies=("tests",),
			),
			Stage(
				"cpu-delta-profiles",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/delta-cpu.json",
					"--updates",
					"96",
					"--changes",
					"128",
					"--pool-size",
					"2048",
					"--queries",
					"1024",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"5",
					"--output",
					str(output / "cpu-delta-profiles"),
				),
				5400,
				dependencies=("tests", "arxiv-delta"),
			),
			Stage(
				"cpu-delta-accuracy",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/delta-accuracy-cpu.json",
					"--updates",
					"4",
					"--warmup",
					"3",
					"--queries",
					"48603",
					"--query-selection",
					"test",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"1",
					"--output",
					str(output / "cpu-delta-accuracy"),
				),
				1200,
				dependencies=("tests", "arxiv-delta"),
			),
		]
	)
	plan.extend(
		[
			Stage(
				"arxiv-row-refresh",
				(
					python,
					"-m",
					"scripts.validate.storage",
					*base,
					"--models",
					"sage",
					"--retain-aggregates",
					"--incremental-aggregates",
					"--row-refresh",
					"--delta-max-steps",
					"2",
					"--batch-size",
					"2048",
					"--worker-bytes",
					str(32 * 1024**2),
					"--output",
					str(output / "arxiv-row-refresh.json"),
				),
				1200,
				dependencies=("tests",),
			),
			Stage(
				"cpu-row-refresh-profiles",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/row-refresh-cpu.json",
					"--updates",
					"96",
					"--changes",
					"128",
					"--pool-size",
					"2048",
					"--queries",
					"1024",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"5",
					"--output",
					str(output / "cpu-row-refresh-profiles"),
				),
				5400,
				dependencies=("tests", "arxiv-row-refresh"),
			),
			Stage(
				"cpu-row-refresh-accuracy",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/row-refresh-accuracy-cpu.json",
					"--updates",
					"4",
					"--warmup",
					"3",
					"--queries",
					"48603",
					"--query-selection",
					"test",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"1",
					"--output",
					str(output / "cpu-row-refresh-accuracy"),
				),
				1200,
				dependencies=("tests", "arxiv-row-refresh"),
			),
			Stage(
				"cpu-row-refresh-drift",
				(
					python,
					"-m",
					"scripts.validate.delta_drift",
					*base,
					"--row-refresh",
					"--refresh-steps",
					"8",
					"--refresh-steps",
					"32",
					"--output",
					str(output / "cpu-row-refresh-drift.json"),
				),
				1200,
				dependencies=("tests", "arxiv-row-refresh"),
			),
		]
	)
	plan.extend(
		[
			Stage(
				"arxiv-aggregate-delta",
				(
					python,
					"-m",
					"scripts.validate.storage",
					*base,
					"--models",
					"sage",
					"--retain-aggregates",
					"--aggregate-only",
					"--incremental-aggregates",
					"--row-refresh",
					"--delta-max-steps",
					"2",
					"--batch-size",
					"2048",
					"--worker-bytes",
					str(32 * 1024**2),
					"--output",
					str(output / "arxiv-aggregate-delta.json"),
				),
				1200,
				dependencies=("tests",),
			),
			Stage(
				"cpu-aggregate-delta-accuracy",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/aggregate-delta-accuracy-cpu.json",
					"--updates",
					"4",
					"--warmup",
					"3",
					"--queries",
					"48603",
					"--query-selection",
					"test",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"1",
					"--output",
					str(output / "cpu-aggregate-delta-accuracy"),
				),
				1800,
				dependencies=("tests", "arxiv-aggregate-delta"),
			),
			Stage(
				"cpu-aggregate-delta-drift",
				(
					python,
					"-m",
					"scripts.validate.delta_drift",
					*base,
					"--aggregate-only",
					"--row-refresh",
					"--refresh-steps",
					"8",
					"--refresh-steps",
					"32",
					"--output",
					str(output / "cpu-aggregate-delta-drift.json"),
				),
				1800,
				dependencies=("tests", "arxiv-aggregate-delta"),
			),
			Stage(
				"cpu-aggregate-delta-profiles",
				(
					python,
					"-m",
					"scripts.evaluate.profile",
					*base,
					"--cases",
					"examples/profiles/aggregate-delta-cpu.json",
					"--updates",
					"96",
					"--changes",
					"128",
					"--pool-size",
					"2048",
					"--queries",
					"1024",
					"--worker-bytes",
					str(32 * 1024**2),
					"--storage-bytes",
					str(256 * 1024**2),
					"--repetitions",
					"5",
					"--output",
					str(output / "cpu-aggregate-delta-profiles"),
				),
				7200,
				dependencies=(
					"tests",
					"arxiv-aggregate-delta",
					"cpu-aggregate-delta-accuracy",
					"cpu-aggregate-delta-drift",
				),
			),
		]
	)
	plan.append(
		Stage(
			"arxiv-storage-probe",
			(
				python,
				"-m",
				"scripts.evaluate.storage",
				"--dataset",
				"ogbn-arxiv",
				"--root",
				str(data_root),
				"--output",
				str(output / "arxiv-storage-probe.json"),
			),
			1200,
			dependencies=("tests", "arxiv-storage"),
		),
	)
	if products:
		plan.append(
			Stage(
				"products-storage",
				(
					python,
					"-m",
					"scripts.evaluate.storage",
					"--dataset",
					"ogbn-products",
					"--root",
					str(data_root),
					"--output",
					str(output / "products-storage.json"),
				),
				2400,
				dependencies=("tests", "arxiv-storage", "arxiv-storage-probe"),
				minimum_host_bytes=32 * 1024**3,
			),
		)
	if products:
		plan.append(
			Stage(
				"products-model",
				(
					python,
					"-m",
					"scripts.validate.storage",
					"--dataset",
					"ogbn-products",
					"--data",
					str(data_root / "products" / "raw"),
					"--models",
					"sage",
					"--changes",
					"4",
					"--batch-size",
					"2048",
					"--worker-bytes",
					str(64 * 1024**2),
					"--storage-bytes",
					str(2 * 1024**3),
					"--output",
					str(output / "products-model.json"),
				),
				3600,
				dependencies=("tests", "arxiv-storage", "products-storage"),
				minimum_host_bytes=32 * 1024**3,
			),
		)
	if products:
		plan.append(
			Stage(
				"products-native",
				(
					python,
					"-m",
					"scripts.evaluate.native_threads",
					"--dataset",
					"ogbn-products",
					"--root",
					str(data_root),
					"--threads",
					"1",
					"--threads",
					"4",
					"--threads",
					"8",
					"--trials",
					"5",
					"--output",
					str(output / "products-native.json"),
				),
				1800,
				dependencies=("arxiv-rpc",),
				minimum_host_bytes=32 * 1024**3,
			)
		)
	return tuple(plan)


def select_devices(requested: Sequence[str], csv_text: str, *, minimum_free_mib: int = 2048) -> Tuple[str, ...]:
	"""Preserve explicit physical order and ignore devices with unknown/free-low memory."""
	eligible = set()
	for row in csv.reader(csv_text.splitlines()):
		if len(row) != 6:
			continue
		try:
			if int(row[3].strip()) >= minimum_free_mib:
				eligible.add(row[0].strip())
		except ValueError:
			continue
	return tuple(index for index in requested if index in eligible)


def _query(command):
	try:
		result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
		return {"returncode": result.returncode, "output": result.stdout}
	except (OSError, subprocess.TimeoutExpired) as error:
		return {"returncode": None, "output": str(error)}


def resources() -> dict:
	return {
		"utc": datetime.now(timezone.utc).isoformat(),
		"host_available_bytes": psutil.virtual_memory().available,
		"gpus": _query(
			[
				"nvidia-smi",
				"--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu",
				"--format=csv,noheader,nounits",
			]
		),
		"gpu_processes": _query(
			[
				"nvidia-smi",
				"--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
				"--format=csv,noheader,nounits",
			]
		),
		"topology": _query(["nvidia-smi", "topo", "-m"]),
		"rdma": _query(["rdma", "link", "show"]),
	}


def _remember(parent: psutil.Process, owned: Dict[int, psutil.Process]) -> None:
	try:
		if not parent.is_running():
			return
		for child in parent.children(recursive=True):
			# Cache creation time now; psutil.is_running checks PID reuse on cleanup.
			child.create_time()
			owned[child.pid] = child
	except psutil.Error:
		pass


def _cleanup(process, parent, owned) -> None:
	_remember(parent, owned)
	if process.poll() is None:
		process.send_signal(signal.SIGSTOP)
		_remember(parent, owned)
	for child in reversed(tuple(owned.values())):
		try:
			if child.is_running():
				child.terminate()
		except psutil.Error:
			pass
	if process.poll() is None:
		process.terminate()
		process.send_signal(signal.SIGCONT)
	_, remaining = psutil.wait_procs(list(owned.values()), timeout=5)
	for child in remaining:
		try:
			if child.is_running():
				child.kill()
		except psutil.Error:
			pass
	try:
		process.wait(timeout=5)
	except subprocess.TimeoutExpired:
		process.kill()
		process.wait(timeout=5)


def run_stage(stage: Stage, folder: Path, environment: dict) -> dict:
	log_path = folder / f"{stage.name}.log"
	start = monotonic()
	owned: Dict[int, psutil.Process] = {}
	with log_path.open("w") as log:
		process = subprocess.Popen(
			stage.command,
			stdout=log,
			stderr=subprocess.STDOUT,
			env={**environment, **dict(stage.environment)},
			start_new_session=True,
		)
		parent = psutil.Process(process.pid)
		parent.create_time()
		try:
			timed_out = False
			while process.poll() is None:
				_remember(parent, owned)
				if monotonic() - start > stage.timeout:
					timed_out = True
					break
				sleep(0.5)
		finally:
			_cleanup(process, parent, owned)
	return {
		"name": stage.name,
		"status": "passed" if process.returncode == 0 and not timed_out else "failed",
		"returncode": process.returncode,
		"timed_out": timed_out,
		"seconds": monotonic() - start,
		"command": stage.command,
		"log": str(log_path),
	}


def unavailable(stage: Stage, outcomes: dict, devices: Sequence[str], context: dict, has_rdma: bool) -> Optional[str]:
	if any(outcomes.get(name, {}).get("status") != "passed" for name in stage.dependencies):
		return "Prerequisite validation did not pass"
	if stage.gpu and not devices:
		return "No requested GPU has at least 2 GiB free; CUDA validation is still required"
	if stage.peers and len(devices) < 2:
		return "Requires two eligible GPUs"
	if stage.rdma and not has_rdma:
		return "No active RDMA link was found"
	if stage.minimum_host_bytes > context["host_available_bytes"]:
		return "Insufficient available host memory for the larger dataset"
	return None


@dataclass
class Options:
	data_root: Path
	output: Path
	gpu_indices: str
	products: bool
	only: list[str] | None
	execute: bool


app = typer.Typer(help=__doc__, no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def main(
	output: Annotated[Path, typer.Option("--output")],
	data_root: Annotated[Path, typer.Option("--data-root")] = Path("data/evaluation"),
	gpu_indices: Annotated[
		str, typer.Option("--gpu-indices", help="Physical NVIDIA indices; only devices with >=2 GiB free are exposed")
	] = "2,3",
	products: Annotated[bool, typer.Option("--products/--no-products")] = True,
	only: Annotated[list[str] | None, typer.Option("--only")] = None,
	execute: Annotated[bool, typer.Option("--execute")] = False,
):
	args = Options(**locals())
	_run(args)


def _run(args: Options) -> None:
	requested = tuple(args.gpu_indices.split(",")) if args.gpu_indices else ()
	if len(set(requested)) != len(requested) or any(not index.isdigit() for index in requested):
		raise typer.BadParameter("GPU indices must be unique comma-separated nonnegative integers")
	plan = stages(args.data_root.resolve(), args.output.resolve(), products=args.products)
	if args.only:
		unknown = set(args.only) - {stage.name for stage in plan}
		if unknown:
			raise typer.BadParameter(f"Unknown stages: {sorted(unknown)}")
		# Include prerequisites so a selective rerun never bypasses correctness gates.
		selected = set(args.only)
		for stage in reversed(plan):
			if stage.name in selected:
				selected.update(stage.dependencies)
		plan = tuple(stage for stage in plan if stage.name in selected)
	if not args.execute:
		print(
			json.dumps(
				{
					"execute": False,
					"requested_gpu_indices": requested,
					"host_operations": [
						"local sockets/process groups",
						"CUDA/NCCL/RDMA when eligible",
						"owned child termination",
						"read existing datasets/checkpoint",
						"write reports/traces under output",
					],
					"installs_or_downloads": False,
					"stages": [asdict(stage) for stage in plan],
				},
				indent=2,
			)
		)
		return
	if args.output.exists() and any(args.output.iterdir()):
		raise typer.BadParameter("Output must be a new or empty directory")
	changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "lib", "scripts", "tests", "pyproject.toml"])
	if changed.returncode:
		raise typer.BadParameter("Commit the checked implementation before executing host validation")
	args.output.mkdir(parents=True, exist_ok=True)
	before = resources()
	devices = select_devices(requested, before["gpus"]["output"])
	environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices), "OMP_NUM_THREADS": "1"}
	context = {
		"before": before,
		"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
		"requested_gpu_indices": requested,
		"effective_gpu_indices": devices,
		"scope": "One host with existing workloads live. Timings do not establish dedicated-cluster performance.",
	}
	(args.output / "resources.json").write_text(json.dumps(context, indent=2) + "\n")
	outcomes: dict[str, dict[str, Any]] = {}
	has_rdma = before["rdma"]["returncode"] == 0 and "ACTIVE" in before["rdma"]["output"].upper()
	for stage in plan:
		current = resources()
		current_devices = select_devices(devices, current["gpus"]["output"])
		reason = unavailable(stage, outcomes, current_devices, current, has_rdma)
		if reason:
			outcome: dict[str, Any] = {"name": stage.name, "status": "skipped", "reason": reason}
		else:
			print(json.dumps({"starting": stage.name, "command": stage.command}), flush=True)
			outcome = run_stage(stage, args.output, {**environment, "CUDA_VISIBLE_DEVICES": ",".join(current_devices)})
		outcome["gpu_indices"] = current_devices
		outcomes[stage.name] = outcome
		(args.output / "results.json").write_text(json.dumps(outcomes, indent=2) + "\n")
		print(json.dumps(outcome), flush=True)
	context["after"] = resources()
	(args.output / "resources.json").write_text(json.dumps(context, indent=2) + "\n")
	if any(
		item["status"] == "failed" or (item["status"] == "skipped" and name not in ("forced-rdma", "products-native"))
		for name, item in outcomes.items()
	):
		raise SystemExit("Required host stages failed or remain unverified; inspect results.json and logs")


if __name__ == "__main__":
	app()
