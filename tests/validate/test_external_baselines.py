import dgl
import pytest
import torch

from benchmarks.external import _classify_failure, capability_manifest
from benchmarks.external_ripple_replay import _full, _measured_replay, external_gcmean_mapping
from benchmarks.inkstream import _assert_simple_graph, _failure_boundary
from lib.models.spec import ModelSpec


def test_external_capabilities_keep_artifacts_separate_from_neutronrt_math():
	ripple = capability_manifest("ripple")
	inkstream = capability_manifest("inkstream")
	assert ripple["exact_math"] == "author GC-mean; not NeutronRT GraphSAGE mean"
	assert ripple["stateful_replay"] and ripple["performance_replay"]
	assert ripple["numerically_verified"]["updates"][-1] == "zero-indegree"
	assert inkstream["author_supported"]["aggregations"] == ["mean", "sum", "max"]
	assert inkstream["adapter_supported"]["aggregations"] == ["mean"]
	assert inkstream["numerically_verified"]["updates"] == ["insert", "delete"]
	assert inkstream["diagnostic_patch"] is None


def test_ripple_gcmean_mapping_requires_zero_self_and_preserves_weight_orientation():
	model = ModelSpec("sage", 4, 8, 3, layers=2).build()
	with pytest.raises(ValueError, match="self terms"):
		external_gcmean_mapping(model)
	for layer in model.layers:
		layer.fc_self.weight.data.zero_()
		layer.fc_self.bias.data.zero_()
	weights = external_gcmean_mapping(model)
	assert [value.shape for value in weights] == [(4, 8), (8, 3)]


@pytest.mark.parametrize("layers", [2, 3])
def test_gcmean_mapping_matches_mean_sage_on_signed_features_and_isolated_node(layers):
	graph = dgl.graph(([0, 1, 2, 3], [1, 2, 0, 1]), num_nodes=5)
	features = torch.tensor(
		[
			[1.0, -2.0, 0.5, -0.25],
			[-0.5, 0.75, -1.25, 2.0],
			[0.25, 1.5, -0.75, -1.0],
			[-1.5, 0.1, 0.2, 0.9],
			[2.0, -0.5, 0.0, -1.0],
		]
	)
	model = ModelSpec("sage", 4, 5, 3, layers=layers).build()
	for layer in model.layers:
		layer.fc_self.weight.data.zero_()
		layer.fc_self.bias.data.zero_()
		if layer.fc_neigh.bias is not None:
			layer.fc_neigh.bias.data.zero_()
	weights = external_gcmean_mapping(model)
	with torch.inference_mode():
		actual = features
		for layer in range(layers):
			actual = model(layer, graph, actual)
	expected = _full(graph, features, weights)
	assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_replay_service_clock_excludes_interleaved_oracle_cost():
	class Processor:
		def __init__(self):
			import numpy as np

			self.embeddings = {1: np.zeros((1, 2), dtype=np.float32)}

		def process_batch(self, batch):
			import time

			time.sleep(0.02)

	measured = [([object()], torch.tensor([0])), ([object()], torch.tensor([0]))]

	def run(oracle_delay):
		processor = Processor()

		def validate_batch(index, batch, current_processor):
			import time

			time.sleep(oracle_delay)
			return True, 0.0

		return _measured_replay(processor, measured, layers=1, validate_batch=validate_batch)

	fast = run(0.0)
	slow = run(0.08)
	assert fast["service_elapsed_seconds"] >= 0.03
	assert slow["loop_wall_seconds"] - slow["service_elapsed_seconds"] >= 0.1
	assert slow["service_elapsed_seconds"] < slow["loop_wall_seconds"]
	assert slow["service_elapsed_seconds"] == pytest.approx(fast["service_elapsed_seconds"], rel=0.4)
	assert 2 / slow["service_elapsed_seconds"] == pytest.approx(2 / fast["service_elapsed_seconds"], rel=0.4)


def test_inkstream_failure_boundary_is_recorded_without_repairing_upstream():
	assert _failure_boundary(0.0, 0.2) == "final_output_stage"
	assert _failure_boundary(0.2, 0.2) == "pre_final_or_hidden"
	assert _failure_boundary(0.0, 0.0) == "none"


def test_inkstream_simple_graph_gate_rejects_multiedges():
	_assert_simple_graph(torch.tensor([[0, 1], [1, 2]]))
	with pytest.raises(ValueError, match="multiedges"):
		_assert_simple_graph(torch.tensor([[0, 0], [1, 1]]))


def test_cuda_runtime_failure_is_not_mislabeled_unsupported():
	assert _classify_failure(1, "CUDA out of memory while running the kernel", None) == "failed"
	assert _classify_failure(1, "CUDA device is not available", None) == "unsupported"
