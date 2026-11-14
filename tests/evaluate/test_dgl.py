import dgl
import pytest
import torch
from typer.testing import CliRunner

from benchmarks.dgl import StreamingRecompute
from benchmarks.streaming import decode_update, run
from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire
from scripts.evaluate import dgl as command


def request(kind, first, second, identity):
	return wire.Update(kind=kind, first=_to_wire(first), second=_to_wire(second), request_id=identity)


def fixture(layers):
	graph = dgl.graph(([0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4]), num_nodes=6)
	features = torch.randn(6, 4, generator=torch.Generator().manual_seed(71))
	spec = ModelSpec("sage", 4, 8, 3, layers=layers)
	updates = [
		request(wire.Update.INSERT, torch.tensor([5]), torch.tensor([1]), "warm-0"),
		request(wire.Update.FEATURES, torch.tensor([0]), torch.full((1, 4), 0.25), "measured-0"),
		request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "measured-1"),
		request(wire.Update.INSERT, torch.tensor([3]), torch.tensor([5]), "measured-2"),
	]
	return graph, features, spec, updates


@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("fanout", [None, 5])
def test_streaming_reference_keeps_all_outputs_correct_after_each_update(layers, fanout):
	graph, features, spec, updates = fixture(layers)
	model = spec.build()
	runner = StreamingRecompute(graph, features, model, fanout=fanout, batch_size=2, output_device="cpu")
	# Lazy initialization must also compute vertices isolated from the first update.
	assert not runner.ready.any()
	for update in updates:
		result = runner.update(decode_update(update))
		assert result["changes"] == 1 and 0 < result["affected_outputs"] <= 6
		with torch.inference_mode():
			expected = features
			for layer in range(layers):
				expected = model(layer, graph, expected)
		torch.testing.assert_close(runner.query(torch.arange(6)), expected, atol=1e-6, rtol=1e-5)
	# The original parallel pair is entirely removed, matching runtime semantics.
	assert not graph.has_edges_between(0, 1)


def test_retained_fused_sampler_observes_each_topology_update(monkeypatch):
	graph, features, spec, updates = fixture(2)
	model = spec.build()
	runner = StreamingRecompute(graph, features, model, fanout=5, batch_size=2, output_device="cpu")
	original = graph.sample_neighbors_fused
	calls = []

	def fused(*args, **kwargs):
		calls.append(True)
		return original(*args, **kwargs)

	monkeypatch.setattr(graph, "sample_neighbors_fused", fused)
	threads = torch.get_num_threads()
	try:
		torch.set_num_threads(2)
		for update in updates:
			runner.update(decode_update(update))
			with torch.inference_mode():
				expected = model(1, graph, model(0, graph, features))
			torch.testing.assert_close(runner.query(torch.arange(6)), expected, atol=1e-6, rtol=1e-5)
	finally:
		torch.set_num_threads(threads)
	assert calls, "The test must exercise DGL's fused sampler"


def recording(graph, features, spec, updates):
	return RecordedWorkload(
		spec,
		inference_identity(graph, features, spec),
		"producer-source",
		tuple(updates[:1]),
		tuple((update, _to_wire(torch.tensor([5, 0, 1]))) for update in updates[1:]),
		torch.arange(6),
		"all",
		71,
		0,
		1e-6,
		1e-5,
	)


@pytest.mark.parametrize("fanout", [None, 5])
def test_replay_includes_output_storage_queries_checks_and_shared_workload_identity(tmp_path, fanout):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	report = run(
		graph,
		features,
		workload,
		tmp_path / "trial",
		fanout=fanout,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert report["valid"] and report["input_identity"] == workload.input_identity
	assert report["trace_digest"] == workload.digest
	assert report["query_latency"]["count"] == 3
	assert len(report["updates"]) == 3 and len(report["correctness"]) == 3
	assert report["output_bytes"] == 6 * 3 * 4
	assert max(item["max_abs_error"] for item in report["correctness"]) < 1e-6
	assert all(item["response_seconds"] >= item["compute_and_store_seconds"] for item in report["updates"])
	assert (tmp_path / "trial/trial.json").exists()


def test_replay_rejects_different_inputs_before_allocating_output(tmp_path):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	features[0] += 1
	with pytest.raises(ValueError, match="differ"):
		run(
			graph,
			features,
			workload,
			tmp_path / "trial",
			fanout=5,
			device=torch.device("cpu"),
			output_device=torch.device("cpu"),
			batch_size=2,
		)
	assert not (tmp_path / "trial").exists()


def test_native_cli_replays_recording_and_keeps_fanout_configurable(tmp_path, monkeypatch, request):
	threads = torch.get_num_threads()
	request.addfinalizer(lambda: torch.set_num_threads(threads))
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	workload.save(tmp_path / "workload.pt")
	graph.ndata["feat"] = features
	dgl.save_graphs(str(tmp_path / "data.bin"), [graph])
	captured = []
	monkeypatch.setattr(command, "run", lambda *args, **kwargs: captured.append((args, kwargs)))
	arguments = [
		"--data",
		str(tmp_path / "data.bin"),
		"--workload",
		str(tmp_path / "workload.pt"),
		"--output",
		str(tmp_path / "trial"),
	]
	result = CliRunner().invoke(command.app, [*arguments, "--fanout", "10"])
	assert result.exit_code == 0, result.output
	assert captured[-1][1]["fanout"] == 10
	assert captured[-1][0][2].digest == workload.digest
	result = CliRunner().invoke(command.app, [*arguments, "--method", "RTEC-Full"])
	assert result.exit_code == 0, result.output
	assert captured[-1][1]["fanout"] is None
	assert CliRunner().invoke(command.app, [*arguments, "--fanout", "0"]).exit_code == 2
	assert CliRunner().invoke(command.app, [*arguments, "--device", "cuda"]).exit_code == 2
