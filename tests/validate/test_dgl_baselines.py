"""Check conventional baselines against independent mean-GraphSAGE equations."""

import os

import dgl
import pytest
import torch

from benchmarks.dgl import infer_batches
from lib.models import GNNModel


@pytest.fixture
def device():
	# Hardware checks run separately; the normal CPU suite never probes CUDA.
	return torch.device(os.environ.get("NEUTRONRT_BASELINE_TEST_DEVICE", "cpu"))


def mean_sage(layer, graph, inputs, *, activate):
	source, destination = graph.edges()
	neighbor_sum = torch.zeros(graph.num_dst_nodes(), inputs.shape[1], device=inputs.device)
	neighbor_sum.index_add_(0, destination, inputs[source])
	mean = neighbor_sum / graph.in_degrees().clamp_min(1).unsqueeze(1)
	result = layer.fc_self(inputs[: graph.num_dst_nodes()]) + layer.fc_neigh(mean)
	return result.relu() if activate else result


@pytest.mark.parametrize("layers", [2, 3])
@pytest.mark.parametrize("fanout", [None, 10])
def test_full_or_exhaustive_sampling_matches_independent_equations_in_requested_order(layers, fanout, device):
	torch.manual_seed(71)
	nodes = torch.arange(16)
	# Include self loops and leave two vertices isolated.
	graph = dgl.graph((nodes.repeat(2), torch.cat((nodes, (nodes + 1) % 16))), num_nodes=18)
	features = torch.randn(18, 4)
	model = GNNModel("sage", 4, 8, 3, layers).eval()
	expected = features
	for index, layer in enumerate(model.layers):
		expected = mean_sage(layer, graph, expected, activate=index + 1 < layers)
	model.to(device)
	requested = torch.tensor([17, 3, 8, 1, 16, 15, 0])
	batches = list(infer_batches(graph, features, model, requested, fanout=fanout, batch_size=3))
	assert torch.equal(torch.cat([ids for ids, _ in batches]), requested)
	actual = torch.cat([value.cpu() for _, value in batches])
	assert not actual.requires_grad
	torch.testing.assert_close(actual, expected[requested], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fanout", [1, 5, 10])
def test_sampled_layers_have_complete_dependencies_and_apply_fanout_everywhere(fanout, device):
	torch.manual_seed(71)
	dgl.seed(71)
	nodes = torch.arange(32)
	graph = dgl.graph((nodes.repeat(16), torch.cat([(nodes + i) % 32 for i in range(16)])), num_nodes=32)
	features = torch.randn(32, 4)

	class ObservedModel(GNNModel):
		previous_ids = None
		previous_values = None

		def forward(self, layer_idx, block, h):
			assert int(block.in_degrees().max()) <= fanout
			if layer_idx:
				assert torch.equal(block.srcdata[dgl.NID], self.previous_ids)
				torch.testing.assert_close(h, self.previous_values)
			actual = super().forward(layer_idx, block, h)
			expected = mean_sage(self.layers[layer_idx], block, h, activate=layer_idx + 1 < self.num_layers)
			torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
			self.previous_ids = block.dstdata[dgl.NID].clone()
			self.previous_values = actual.clone()
			return actual

	model = ObservedModel("sage", 4, 8, 3, 3).eval().to(device)
	batches = list(infer_batches(graph, features, model, torch.arange(32), fanout=fanout, batch_size=7))
	assert torch.cat([value for _, value in batches]).shape == (32, 3)


@pytest.mark.parametrize("fanout", [0, -1, True])
def test_invalid_sampling_fanout_is_rejected(fanout):
	graph = dgl.graph((torch.tensor([0]), torch.tensor([0])))
	model = GNNModel("sage", 4, 8, 3, 2).eval()
	with pytest.raises(ValueError, match="fanout"):
		list(infer_batches(graph, torch.ones(1, 4), model, torch.tensor([0]), fanout=fanout, batch_size=1))
