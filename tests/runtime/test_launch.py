import dgl
import pytest
import torch
import torch.multiprocessing as mp
from typer.testing import CliRunner

from lib.graph.ownership import Owner
from lib.models.spec import ModelSpec
from lib.runtime.config import RuntimeFeatures
from lib.runtime.launch import Placement, WorkerSettings, resolve_devices, run_worker
from lib.runtime.managed import ManagedRuntime
from lib.runtime.transport import RemoteEndpoint
from scripts import simple


def test_placement_uses_launcher_values_and_explicit_overrides():
	env = {"RANK": "3", "WORLD_SIZE": "6", "LOCAL_RANK": "1"}
	assert Placement.resolve(environ=env) == Placement(3, 6, 1)
	assert Placement.resolve(rank=2, world_size=4, local_rank=0, environ=env) == Placement(2, 4, 0)
	assert Placement.resolve(environ={"WORLD_SIZE": "2"}).rank is None
	with pytest.raises(ValueError):
		Placement.resolve(rank=3, world_size=2, environ={})


def test_device_selection_accounts_for_storage_slot_and_visible_devices(monkeypatch):
	monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
	assert resolve_devices() == ("cuda:0", "cuda:1")
	assert Placement(1, 4, 1).worker_device() == "cuda:0"
	assert Placement(2, 4, 0).worker_device() == "cuda:0"
	assert Placement(3, 4, 1).worker_device(["cuda:1", "cuda:0"]) == "cuda:0"
	assert Placement(1, 2, 1).worker_device(device="cpu") == "cpu"
	with pytest.raises(ValueError, match="visible"):
		resolve_devices(["cuda:2"])
	monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
	assert resolve_devices() == ("cpu",)


def test_cli_preserves_launcher_environment_and_has_no_machine_specific_path(tmp_path, monkeypatch):
	config = tmp_path / "env.json"
	config.write_text('{"WORLD_SIZE": 9, "MASTER_ADDR": "storage.example", "MASTER_RPC_PORT": 12345}')
	monkeypatch.setenv("RANK", "1")
	monkeypatch.setenv("WORLD_SIZE", "3")
	monkeypatch.delenv("MASTER_ADDR", raising=False)
	monkeypatch.delenv("MASTER_RPC_PORT", raising=False)
	captured = {}
	monkeypatch.setattr(simple, "_run", lambda args, placement: captured.update(args=args, placement=placement))
	result = CliRunner().invoke(
		simple.app,
		[
			"--env",
			str(config),
			"--threads",
			"2",
			"--history-bytes",
			"1024",
			"--storage-overlay-bytes",
			"1073741824",
			"--storage-response-bytes",
			"67108864",
		],
	)
	assert result.exit_code == 0, result.output
	args, placement = captured["args"], captured["placement"]
	assert placement.world_size == 3 and placement.rank == 1
	assert args.listen == "storage.example:12345"
	assert args.topology_root is None and args.profile_report is None
	assert args.history_bytes == 1024 and args.threads == 2
	assert args.storage_overlay_bytes == 1024**3 and args.storage_response_bytes == 64 * 1024**2


def _external_rank(rank, rendezvous):
	torch.set_num_threads(1)
	placement = Placement(rank, 3, rank, rendezvous)
	switches = RuntimeFeatures(operator_incremental=True)
	if rank:
		run_worker(
			None,
			RemoteEndpoint(Owner(rank, 1), torch.device("cpu"), rank * 1024**2),
			3,
			rendezvous,
			WorkerSettings(switches, threads=1),
		)
		return
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=4)
	features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
	spec = ModelSpec("gat", 4, 5, 3)
	model = spec.build()
	with ManagedRuntime(
		graph, features, spec, placement=placement, switches=switches, worker_threads=1, batch_size=2
	) as runtime:
		assert not runtime.processes
		assert sorted(endpoint.max_bytes for endpoint in runtime.plane.remotes.values()) == [1024**2, 2 * 1024**2]
		for step in range(2):
			if step:
				runtime.service.change_features(torch.tensor([0]), features[[0]] + 0.2)
			nodes = torch.arange(4)
			runtime.service.ensure(nodes)
			with runtime.service.query(nodes, runtime.plane.storage.owner) as actual:
				torch.testing.assert_close(actual, model(1, graph, model(0, graph, features)), atol=2e-5, rtol=2e-5)


@pytest.mark.host
def test_externally_placed_storage_and_workers_share_the_serving_protocol(tmp_path):
	mp.spawn(_external_rank, args=(f"file://{tmp_path / 'world'}",), nprocs=3, join=True)


@pytest.mark.host
def test_torchrun_cli_resolves_roles_without_workers_loading_graph_files(tmp_path):
	import gzip
	import json
	import os
	import signal
	import subprocess
	import sys
	import time

	import grpc

	from lib.rpc import inference_pb2 as wire
	from lib.rpc.inference_pb2_grpc import OwnedInferenceStub
	from lib.rpc.service import _from_wire as from_wire
	from lib.rpc.service import _to_wire as to_wire

	raw = tmp_path / "raw"
	raw.mkdir()
	features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
	files = {
		"num-node-list": "4\n",
		"num-edge-list": "4\n",
		"edge": "0,1\n0,1\n1,2\n2,0\n",
		"node-feat": "\n".join(",".join(str(value) for value in row) for row in features.tolist()) + "\n",
	}
	for name, contents in files.items():
		with gzip.open(raw / f"{name}.csv.gz", "wt") as stream:
			stream.write(contents)
	ready_path = tmp_path / "ready.json"
	command = [
		sys.executable,
		"-m",
		"torch.distributed.run",
		"--standalone",
		"--nproc-per-node=3",
		"-m",
		"scripts.simple",
		"--data",
		str(raw),
		"--device",
		"cpu",
		"--classes",
		"3",
		"--hidden",
		"5",
		"--operator-incremental",
		"--threads",
		"1",
		"--listen",
		"127.0.0.1:0",
		"--ready-file",
		str(ready_path),
	]
	with (tmp_path / "launcher.log").open("w+") as output:
		process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
		try:
			deadline = time.monotonic() + 60
			while not ready_path.exists():
				if process.poll() is not None or time.monotonic() > deadline:
					output.seek(0)
					pytest.fail(output.read())
				time.sleep(0.05)
			ready = json.loads(ready_path.read_text())
			with grpc.insecure_channel(ready["address"]) as channel:
				response = OwnedInferenceStub(channel).Query(
					wire.QueryRequest(nodes=to_wire(torch.arange(4))), timeout=20
				)
				model = ModelSpec("sage", 4, 5, 3).build()
				graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=4)
				with torch.inference_mode():
					expected = model(1, graph, model(0, graph, features))
				torch.testing.assert_close(
					from_wire(response.values, len(response.values.data)), expected, atol=2e-5, rtol=2e-5
				)
			from scripts.benchmark import probe

			report_path = tmp_path / "benchmark.json"
			probe(ready["address"], report_path, requests=2, warmup=0, query_sizes=[1, 3], updates=2, changes=1)
			report = json.loads(report_path.read_text())
			assert report["api"] == "inference"
			assert [entry["nodes"] for entry in report["queries"]] == [1, 3]
			assert len(report["updates"]["completion_seconds"]) == 2
			os.kill(ready["pid"], signal.SIGTERM)
			assert process.wait(timeout=20) == 0
		finally:
			if process.poll() is None:
				process.terminate()
				try:
					process.wait(timeout=10)
				except subprocess.TimeoutExpired:
					process.kill()
					process.wait()
