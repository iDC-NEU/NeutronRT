import pytest
import torch

from lib.models.spec import ModelSpec


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
def test_replacement_weights_match_without_changing_the_master_random_stream(name):
	spec = ModelSpec(name, 4, 5, 3, seed=19)
	before = torch.random.get_rng_state()
	first, replacement = spec.build(), spec.build()
	assert torch.equal(before, torch.random.get_rng_state())
	assert spec.widths == (4, 5, 3)
	for key, value in first.state_dict().items():
		torch.testing.assert_close(value, replacement.state_dict()[key], rtol=0, atol=0)
	assert not first.training and next(first.parameters()).device.type == "cpu"


def test_replacement_loads_the_same_checkpoint_and_rejects_another_configuration(tmp_path):
	spec = ModelSpec("sage", 4, 5, 3, seed=19)
	path = tmp_path / "checkpoint.pt"
	torch.save(
		{
			"model_config": {
				"model_name": "sage",
				"num_features": 4,
				"num_hiddens": 5,
				"num_classes": 3,
				"num_layers": 2,
			},
			"state_dict": spec.build().state_dict(),
		},
		path,
	)
	restored = ModelSpec("sage", 4, 5, 3, seed=77, checkpoint=str(path)).build()
	for key, value in spec.build().state_dict().items():
		torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
	with pytest.raises(ValueError, match="configuration"):
		ModelSpec("sage", 4, 6, 3, checkpoint=str(path)).build()
	resolved = ModelSpec.from_options(4, checkpoint=str(path))
	assert resolved.name == "sage" and resolved.hidden == 5 and resolved.classes == 3
	assert ModelSpec.from_options(4, checkpoint=str(path), hidden=9).hidden == 9
	with pytest.raises(ValueError, match="--classes"):
		ModelSpec.from_options(4)
