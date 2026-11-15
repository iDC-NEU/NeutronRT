from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dgl
import pytest
import torch
from typer.testing import CliRunner

from lib.models.spec import ModelSpec
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _from_wire, _to_wire
from scripts.evaluate import distributed


def test_default_methods_start_with_incremental_and_leave_full_last(monkeypatch, tmp_path):
	captured = []
	monkeypatch.setattr(distributed, "_run", captured.append)
	result = CliRunner().invoke(distributed.app, ["--output", str(tmp_path)])
	assert result.exit_code == 0, result.output
	assert captured[0].methods == ["NrtInc", "RTEC-NS", "RTEC-UER", "RTEC-Full"]
	assert captured[0].fanout == 5


@pytest.mark.parametrize("methods", [["NrtInc", "RTEC-NS", "RTEC-UER", "RTEC-Full"], ["RTEC-UER", "NrtInc"]])
def test_experiment_finishes_method_repetitions_before_starting_next_method(monkeypatch, methods):
	seen = []
	monkeypatch.setattr(distributed.Placement, "resolve", lambda **kwargs: SimpleNamespace(rank=1))
	monkeypatch.setattr(distributed.torch, "set_num_threads", lambda count: None)
	monkeypatch.setattr(
		distributed,
		"trial",
		lambda args, placement, graph, features, model, method, repetition: seen.append((repetition, method)),
	)
	args = SimpleNamespace(
		rank=1,
		world_size=2,
		local_rank=0,
		init_method=None,
		threads=1,
		repetitions=2,
		models=["sage"],
		methods=methods,
	)
	distributed._run(args)
	assert seen == [(repetition, method) for method in methods for repetition in range(2)]


def test_staged_trace_preserves_repeated_feature_changes_without_mutating_input():
	graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
	features = torch.zeros(2, 4)
	staged = features.clone()
	pending = {}
	generator = torch.Generator().manual_seed(71)
	for index in (0, 3, 6):
		oracle_generator = torch.Generator().set_state(generator.get_state())
		expected_nodes = distributed.sample_ids(2, 2, oracle_generator)
		expected_values = staged[expected_nodes] + torch.randn(2, 4, generator=oracle_generator) * 0.05
		request = distributed.make_update(
			graph,
			features,
			ModelSpec("sage", 4, 4, 2),
			index,
			2,
			generator,
			str(index),
			workload="mixed",
			pending=pending,
		)
		nodes = _from_wire(request.first, 1024)
		values = _from_wire(request.second, 1024)
		torch.testing.assert_close(nodes, expected_nodes)
		torch.testing.assert_close(values, expected_values)
		staged[nodes] = values
		assert torch.equal(features, torch.zeros_like(features))
		assert all(torch.equal(staged[node], value) for node, value in pending.items())
	assert not torch.equal(staged, features)


def test_staged_topology_changes_leave_runtime_graph_untouched():
	graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
	oracle = graph.clone()
	request = wire.Update(
		kind=wire.Update.INSERT, first=_to_wire(torch.tensor([2])), second=_to_wire(torch.tensor([0]))
	)
	distributed.mirror_edges(oracle, request)
	assert graph.num_edges() == 2 and oracle.num_edges() == 3
	request.kind = wire.Update.DELETE
	distributed.mirror_edges(oracle, request)
	assert oracle.num_edges() == graph.num_edges() == 2


def test_edge_change_record_separates_requested_pairs_from_effective_delta():
	graph = dgl.graph(([0], [1]), num_nodes=2)
	request = wire.Update(
		kind=wire.Update.DELETE, first=_to_wire(torch.tensor([0, 0])), second=_to_wire(torch.tensor([1, 1]))
	)
	record = distributed.edge_change_record(graph, request, mirror=True)
	assert record["requested_changes"] == 2
	assert record["effective_edge_delta"] == -1
	assert record["effective_edge_changes"] == 1


def test_cli_accepts_canonical_methods_and_rejects_duplicates(monkeypatch, tmp_path):
	captured = []
	monkeypatch.setattr(distributed, "_run", captured.append)
	command = [
		"--output",
		str(tmp_path),
		"--methods",
		"RTEC-UER",
		"--methods",
		"NrtInc",
		"--storage-overlay-bytes",
		"1073741824",
		"--storage-response-bytes",
		"67108864",
		"--initialization-batch-size",
		"32",
	]
	result = CliRunner().invoke(distributed.app, command)
	assert result.exit_code == 0, result.output
	assert captured[0].methods == ["RTEC-UER", "NrtInc"]
	assert captured[0].storage_overlay_bytes == 1024**3 and captured[0].storage_response_bytes == 64 * 1024**2
	assert captured[0].initialization_batch_size == 32
	result = CliRunner().invoke(
		distributed.app, ["--output", str(tmp_path), "--methods", "RTEC-UER", "--methods", "RTEC-UER"]
	)
	assert result.exit_code == 2


@pytest.mark.parametrize("workload", ["edges", "mixed"])
def test_trials_isolate_topology_caches_and_copy_features_only_for_mutable_traces(monkeypatch, tmp_path, workload):
	from lib.runtime.launch import Placement

	captured = []
	monkeypatch.setattr(distributed, "_run", captured.append)
	result = CliRunner().invoke(
		distributed.app,
		["--output", str(tmp_path), "--models", "sage", "--methods", "RTEC-NS", "--workload", workload],
	)
	assert result.exit_code == 0, result.output
	args = captured[0]
	args.source_fingerprint = "unit-test-source"
	graph = dgl.graph(([0, 1, 1], [1, 0, 2]), num_nodes=4)
	features = torch.ones(4, 4)
	formats = graph.formats()

	class StopAtRuntime(Exception):
		pass

	def inspect_inputs(trial_graph, trial_features, *positional, **keywords):
		trial_graph.create_formats_()
		trial_graph.edata["trial_only"] = torch.ones(trial_graph.num_edges())
		assert graph.formats() == formats and "trial_only" not in graph.edata
		assert (trial_features.data_ptr() == features.data_ptr()) == (workload == "edges")
		if workload == "mixed":
			trial_features.add_(1)
			torch.testing.assert_close(features, torch.ones_like(features))
		raise StopAtRuntime

	monkeypatch.setattr(distributed, "ManagedRuntime", inspect_inputs)
	with pytest.raises(StopAtRuntime):
		distributed.trial(args, Placement(0, 2, 0, "unused-rendezvous"), graph, features, "sage", "RTEC-NS", 0)


@pytest.mark.parametrize("fanout", [1, 5, 10])
def test_sampling_fanout_is_an_explicit_typer_option(monkeypatch, tmp_path, fanout):
	captured = []
	monkeypatch.setattr(distributed, "_run", captured.append)
	result = CliRunner().invoke(
		distributed.app, ["--output", str(tmp_path), "--methods", "RTEC-NS", "--fanout", str(fanout)]
	)
	assert result.exit_code == 0, result.output
	assert captured[0].fanout == fanout and captured[0].methods == ["RTEC-NS"]
	result = CliRunner().invoke(distributed.app, ["--output", str(tmp_path), "--methods", "RTEC-NS", "--fanout", "0"])
	assert result.exit_code == 2


def test_neighborhood_sampling_preserves_message_passing_edges_and_is_repeatable(monkeypatch):
	generator = torch.Generator().manual_seed(9)
	order = torch.randperm(100, generator=generator)
	graph = dgl.graph((order[:-1], order[1:]), num_nodes=100)
	graph.ndata["feat"] = torch.arange(400, dtype=torch.float32).reshape(100, 4)
	monkeypatch.setattr(distributed, "load_graph_data", lambda *args, **kwargs: graph.clone())
	args = SimpleNamespace(
		seed=71, data=Path("unused-by-test"), node_limit=10, node_selection="neighbors", add_inverse_edges=False
	)
	first, features = distributed.load_graph(args)
	second, repeated = distributed.load_graph(args)
	assert first.num_nodes() == 10 and first.num_edges() == 9
	assert torch.equal(first.ndata[dgl.NID], second.ndata[dgl.NID])
	assert torch.equal(features, repeated)


@pytest.mark.parametrize("population,count", [(0, 0), (1000000000, 7), (100, 0), (17, 17), (17, 5)])
def test_sparse_trace_sampling_is_bounded_distinct_and_repeatable(population, count, monkeypatch):
	original = torch.randperm

	def bounded_randperm(size, **kwargs):
		assert size <= max(32, 2 * count)
		return original(size, **kwargs)

	monkeypatch.setattr(torch, "randperm", bounded_randperm)
	first = distributed.sample_ids(population, count, torch.Generator().manual_seed(71))
	second = distributed.sample_ids(population, count, torch.Generator().manual_seed(71))
	assert torch.equal(first, second)
	assert first.numel() == first.unique().numel() == count
	assert not count or bool(((first >= 0) & (first < population)).all())


def test_verification_chunks_rpc_outputs_and_rejects_mixed_versions():
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	model = ModelSpec("sage", 4, 6, 3).build()
	with torch.inference_mode():
		expected = model(1, graph, model(0, graph, features))
	versions = iter([7, 7, 7])
	seen = []

	def query(request, **kwargs):
		nodes = _from_wire(request.nodes, 1024)
		seen.append(nodes)
		assert nodes.numel() <= 2
		return wire.QueryResponse(values=_to_wire(expected[nodes]), version=next(versions))

	stub = SimpleNamespace(Query=Mock(side_effect=query))
	report = distributed.verify(stub, model, graph, features, batch_size=2)
	assert report["rows"] == 5 and report["oracle_scope"] == "all rows"
	assert report["mean_squared_error"] == 0
	assert torch.equal(torch.cat(seen), torch.arange(5))
	versions = iter([7, 8, 8])
	with pytest.raises(RuntimeError, match="version changed"):
		distributed.verify(stub, model, graph, features, batch_size=2)
