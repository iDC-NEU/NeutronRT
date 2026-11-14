import dgl
import pytest
import torch

from lib.models import GNNModel


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("custom_postprocessing", [False, True])
@pytest.mark.parametrize(
	"device",
	["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"))],
)
def test_external_mean_transform_matches_graphsage_layer(layer, custom_postprocessing, device):
	torch.manual_seed(17)
	model = GNNModel("sage", 5, 4, 3, 2, aggregator_type="mean").eval().to(device)
	if custom_postprocessing:
		model.layers[layer].activation = torch.sigmoid
		model.layers[layer].norm = torch.nn.functional.normalize
	u = torch.tensor([0, 0, 1, 2, 4], device=device)
	v = torch.tensor([1, 1, 1, 2, 2], device=device)
	block = dgl.create_block((u, v), num_src_nodes=6, num_dst_nodes=4, device=device)
	width = 5 if layer == 0 else 4
	values = torch.randn(6, width, device=device)
	mean = torch.zeros(4, width, device=device)
	mean.index_add_(0, v, values[u])
	mean /= block.in_degrees().clamp_min(1).unsqueeze(1)
	with torch.inference_mode():
		actual = model.from_aggrs(layer, values[:4], mean)
		expected = model(layer, block, values)
	torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_mean_transform_rejects_training_and_other_aggregators():
	model = GNNModel("sage", 3, 4, 2, 2, aggregator_type="mean")
	with pytest.raises(ValueError, match="eval"):
		model.from_aggrs(0, torch.ones(2, 3), torch.ones(2, 3))
	model = GNNModel("sage", 3, 4, 2, 2, aggregator_type="pool").eval()
	with pytest.raises(NotImplementedError, match="mean"):
		model.from_aggrs(0, torch.ones(2, 3), torch.ones(2, 3))


def test_unknown_model_name_reports_a_configuration_error():
	with pytest.raises(ValueError, match="Unknown GNN model"):
		GNNModel("missing", 3, 4, 2, 2)
