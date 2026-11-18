import inspect
import json

import dgl
import pytest
import torch
from typer.testing import CliRunner

import benchmarks.dgl as dgl_reference
import benchmarks.streaming as streaming
from benchmarks.dgl import StreamingRecompute, UERRecompute
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


def test_delete_absent_pair_is_noop_and_delete_removes_all_parallel_edges():
	graph = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([1, 1, 2, 2])), num_nodes=3)
	features = torch.randn(3, 4, generator=torch.Generator().manual_seed(71))
	spec = ModelSpec("sage", 4, 8, 3, layers=2)
	runner = StreamingRecompute(graph, features, spec.build(), fanout=None, batch_size=2, output_device="cpu")
	runner.initialize()
	before = graph.num_edges()
	absent = decode_update(request(wire.Update.DELETE, torch.tensor([2]), torch.tensor([0]), "absent"))
	result = runner.update(absent)
	assert graph.num_edges() == before and result["sampler_generation"] == 0
	parallel = decode_update(request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "parallel"))
	runner.update(parallel)
	assert graph.num_edges() == 2 and not graph.has_edges_between(0, 1)
	assert runner.sampler_generation == 1


def test_uer_reuses_layerwise_state_with_real_index_writeback_across_updates():
	graph, features, spec, _ = fixture(2)
	runner = UERRecompute(graph, features, spec.build(), fanout=None, batch_size=2, output_device="cpu")
	query = torch.tensor([5, 2, 0, 4], dtype=torch.int64)
	assert runner.query(query).shape == (4, 3)
	initial_state = runner.states[1].clone()

	updates = [
		decode_update(request(wire.Update.FEATURES, torch.tensor([4, 1]), torch.full((2, 4), 0.5), "uer-0")),
		decode_update(request(wire.Update.DELETE, torch.tensor([2]), torch.tensor([0]), "uer-absent")),
		decode_update(request(wire.Update.INSERT, torch.tensor([5]), torch.tensor([2]), "uer-1")),
		decode_update(request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "uer-2")),
	]
	before_absent = graph.num_edges()
	for update in updates:
		result = runner.update(update)
		with torch.inference_mode():
			expected = features
			for layer in range(spec.layers):
				expected = runner.model(layer, graph, expected)
		torch.testing.assert_close(runner.query(query), expected[query], atol=1e-6, rtol=1e-5)
		assert result["publication"]["status"] in {
			"standalone_final_output_index_copy",
			"no_op_absent_delete",
		}
		assert len(result["affected_by_layer"]) == spec.layers
		if update.kind == "delete" and update.first.tolist() == [2]:
			assert result["affected_outputs"] == 0
			assert graph.num_edges() == before_absent
	assert runner.sampler_generation == 2
	assert not torch.equal(initial_state, runner.states[1])


@pytest.mark.parametrize("layers", [2, 3])
def test_uer_keeps_edge_destination_self_dependency_without_self_loops(layers):
	graph = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])), num_nodes=4)
	features = torch.randn(4, 4, generator=torch.Generator().manual_seed(71))
	spec = ModelSpec("sage", 4, 8, 3, layers=layers)
	runner = UERRecompute(graph, features, spec.build(), fanout=None, batch_size=2, output_device="cpu")
	runner.initialize()
	for update in (
		decode_update(request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "chain-delete")),
		decode_update(request(wire.Update.INSERT, torch.tensor([0]), torch.tensor([1]), "chain-insert")),
	):
		result = runner.update(update)
		with torch.inference_mode():
			expected = features
			for layer in range(layers):
				expected = runner.model(layer, graph, expected)
		actual = runner.query(torch.arange(4))
		torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
		assert 1 in runner._affected_sets(update)[0].tolist()
		assert result["affected_by_layer"][0] >= 1
		assert result["affected_by_layer"][-1] >= 1
	# The isolated node remains queryable and is not forced into the affected set.
	assert (
		3
		not in runner._affected_sets(
			decode_update(request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "isolated-check"))
		)[-1].tolist()
	)


@pytest.mark.parametrize("layers", [2, 3])
def test_uer_affected_sets_are_minimal_for_edge_and_feature_updates(layers):
	graph = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])), num_nodes=4)
	features = torch.randn(4, 4, generator=torch.Generator().manual_seed(7))
	spec = ModelSpec("sage", 4, 8, 3, layers=layers)
	runner = UERRecompute(graph, features, spec.build(), fanout=None, batch_size=2, output_device="cpu")
	edge = decode_update(request(wire.Update.INSERT, torch.tensor([0]), torch.tensor([1]), "edge"))
	feature = decode_update(request(wire.Update.FEATURES, torch.tensor([0]), torch.ones(1, 4), "feature"))
	edge_expected = [{1}, {1, 2}] if layers == 2 else [{1}, {1, 2}, {1, 2, 3}]
	feature_expected = [{0, 1}, {0, 1, 2}] if layers == 2 else [{0, 1}, {0, 1, 2}, {0, 1, 2, 3}]
	assert [set(row.tolist()) for row in runner._affected_sets(edge)] == edge_expected
	assert [set(row.tolist()) for row in runner._affected_sets(feature)] == feature_expected


def test_all_four_methods_call_the_shared_authoritative_mutator(monkeypatch):
	from lib.graph import resident_runtime

	graph, features, spec, updates = fixture(2)
	original = dgl_reference.apply_prepared_mutation
	calls = []

	def observe(*args, **kwargs):
		calls.append(True)
		return original(*args, **kwargs)

	monkeypatch.setattr(dgl_reference, "apply_prepared_mutation", observe)
	monkeypatch.setattr(resident_runtime, "apply_prepared_mutation", observe)
	resident = resident_runtime.ResidentGraphSAGE(
		graph.clone(), features.clone(), spec.build(), device="cpu", capacity_bytes=64 * 1024**2
	)
	resident.bootstrap()
	resident.update(decode_update(updates[1]))
	for runner_type, fanout in ((StreamingRecompute, None), (StreamingRecompute, 5), (UERRecompute, None)):
		runner = runner_type(
			graph.clone(), features.clone(), spec.build(), fanout=fanout, batch_size=2, output_device="cpu"
		)
		runner.update(decode_update(updates[1]))
	assert len(calls) == 4


def test_uer_rejects_sampled_fanout_and_replay_requires_exact_full_neighbor(tmp_path):
	graph, features, spec, updates = fixture(2)
	with pytest.raises(ValueError, match="exact full-neighbor"):
		UERRecompute(graph, features, spec.build(), fanout=5, batch_size=2, output_device="cpu")
	workload = recording(graph, features, spec, updates)
	with pytest.raises(ValueError, match="exact full-neighbor"):
		run(
			graph,
			features,
			workload,
			tmp_path / "uer-sampled",
			fanout=5,
			device=torch.device("cpu"),
			output_device=torch.device("cpu"),
			batch_size=2,
			method="RTEC-UER",
		)


def test_replay_oracle_mutation_is_outside_measured_window(tmp_path, monkeypatch):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates[:2])
	clock_value = 0.0

	def clock():
		nonlocal clock_value
		clock_value += 1.0
		return clock_value

	original_apply = streaming.apply_update

	def expensive_oracle_apply(graph_, features_, request_):
		nonlocal clock_value
		clock_value += 100.0
		return original_apply(graph_, features_, request_)

	monkeypatch.setattr(streaming, "perf_counter", clock)
	monkeypatch.setattr(streaming, "apply_update", expensive_oracle_apply)
	report = streaming.run(
		graph,
		features,
		workload,
		tmp_path / "trial",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert report["window_seconds"] < 10.0
	assert report["environment"]["memory"]["oracle_clone_physical_bytes"] is None
	assert "harness_inclusive" in next(
		key for key in report["environment"]["memory"] if key.startswith("host_peak_rss")
	)


def test_initialization_seconds_stop_before_oracle_check(tmp_path, monkeypatch):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	clock_value = 0.0

	def clock():
		nonlocal clock_value
		clock_value += 1.0
		return clock_value

	original_error = streaming.query_error

	def expensive_check(actual, expected, **kwargs):
		nonlocal clock_value
		clock_value += 100.0
		return original_error(actual, expected, **kwargs)

	monkeypatch.setattr(streaming, "perf_counter", clock)
	monkeypatch.setattr(streaming, "query_error", expensive_check)
	report = streaming.run(
		graph,
		features,
		workload,
		tmp_path / "timed-init",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert report["initialization"]["seconds"] < 10.0


def test_logical_clone_bytes_do_not_materialize_graph_edge_arrays():
	graph, features, _, _ = fixture(2)
	assert ".edges(" not in inspect.getsource(streaming._logical_clone_bytes)
	assert streaming._logical_clone_bytes(graph, features)["edge_id_tensor_bytes"] == graph.num_edges() * 8


def test_full_updates_use_native_full_sampler_and_report_sampling_identity(tmp_path):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	report = run(
		graph,
		features,
		workload,
		tmp_path / "trial",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert report["sampling"]["sampler"] == "MultiLayerFullNeighborSampler"
	assert report["backend"]["upstream_commit"]
	assert report["final_state_validation"]["edges_match"]
	assert report["final_state_validation"]["features_match"]
	assert report["final_oracle"]["rows"] == graph.num_nodes()
	assert report["final_quality"]["rows"] == graph.num_nodes()
	assert report["initialization"]["timing"]["native_sampler_block_seconds"] > 0
	assert report["updates"][0]["publication"]["status"] == "not_applicable_standalone"


def test_uer_replay_is_explicit_author_adaptation(tmp_path):
	graph, features, spec, updates = fixture(2)
	workload = recording(graph, features, spec, updates)
	report = run(
		graph,
		features,
		workload,
		tmp_path / "uer-trial",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
		method="RTEC-UER",
		phase_file=tmp_path / "uer-phase.json",
	)
	assert report["implementation"] == "DGL-RTEC-UER"
	assert report["comparison_group"] == "framework_ablation"
	assert report["backend_role"] == "author_layerwise_uer_adaptation"
	assert report["backend"]["uer_provenance"]["official_dgl_uer"] is False
	assert report["timing_semantics"]["publication"] == "standalone final output index_copy publication"
	assert report["fanout"] is None
	assert (
		report["environment"]["memory"]["retained_intermediate_state_bytes"]
		< report["environment"]["memory"]["retained_intermediate_state_total_referenced_bytes"]
	)
	phase = json.loads((tmp_path / "uer-phase.json").read_text())
	assert phase["phase"] == "done"
	assert phase["method"] == "RTEC-UER"
	assert phase["repetition"] == workload.repetition
	assert phase["trace_digest"] == workload.digest
	assert phase["backend_identity"].startswith("dgl-2.4.0-")


def test_output_cache_budget_fails_closed_before_allocation():
	graph, features, spec, _ = fixture(2)
	with pytest.raises(ValueError, match="Output cache needs"):
		StreamingRecompute(
			graph,
			features,
			spec.build(),
			fanout=None,
			batch_size=2,
			output_device="cpu",
			output_budget_bytes=1,
		)
	with pytest.raises(ValueError, match="CUDA output caches require"):
		StreamingRecompute(
			graph,
			features,
			spec.build(),
			fanout=None,
			batch_size=2,
			output_device="cuda",
		)


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


@pytest.mark.parametrize("method", ["RTEC-Full", "RTEC-NS", "RTEC-UER"])
def test_oracle_mutations_are_deferred_until_service_trace_finishes(tmp_path, monkeypatch, method):
	graph, features, spec, updates = fixture(2)
	order = []
	runner_type = UERRecompute if method == "RTEC-UER" else StreamingRecompute
	original_update = runner_type.update
	original_oracle = streaming.apply_update

	def service(self, request):
		order.append("service")
		return original_update(self, request)

	def oracle(*args):
		order.append("oracle")
		return original_oracle(*args)

	monkeypatch.setattr(runner_type, "update", service)
	monkeypatch.setattr(streaming, "apply_update", oracle)
	run(
		graph,
		features,
		recording(graph, features, spec, updates),
		tmp_path / method,
		fanout=5 if method == "RTEC-NS" else None,
		method=method,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert order == ["service", "oracle"] + ["service"] * 3 + ["oracle"] * 3


def test_standalone_replay_reports_labelled_accuracy_outside_timing(tmp_path):
	graph, features, spec, updates = fixture(2)
	report = run(
		graph,
		features,
		recording(graph, features, spec, updates),
		tmp_path / "labelled",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
		labels=torch.tensor([0, 1, 2, 0, 1, 2]),
		test_mask=torch.tensor([True, False, True, False, True, False]),
	)
	assert all(check["labelled_test_rows"] == 3 for check in report["correctness"])
	assert all(check["accuracy_loss_percentage_points"] == 0 for check in report["correctness"])


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
	assert all(abs(sum(item["phase_seconds"].values()) - item["response_seconds"]) < 1e-6 for item in report["updates"])
	assert all(item.get("query_seconds", 0.0) >= 0 for item in report["updates"])
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
	result = CliRunner().invoke(command.app, [*arguments, "--method", "RTEC-UER", "--fanout", "10"])
	assert result.exit_code == 0, result.output
	assert captured[-1][1]["method"] == "RTEC-UER"
	assert captured[-1][1]["fanout"] is None
	assert CliRunner().invoke(command.app, [*arguments, "--fanout", "0"]).exit_code == 2
	assert CliRunner().invoke(command.app, [*arguments, "--device", "cuda"]).exit_code == 2
