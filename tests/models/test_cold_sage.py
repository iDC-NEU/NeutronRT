from dataclasses import replace

import dgl
import pytest
import torch

from lib.graph.affected import AffectedPlanner
from lib.models.incremental import IncrementalLayers
from lib.models.spec import ModelSpec


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@torch.inference_mode()
def test_cold_batch_retains_signed_state_with_postprocessing_and_parallel_edges(device):
	if device.startswith("cuda") and not torch.cuda.is_available():
		pytest.skip("CUDA is unavailable")
	torch.set_num_threads(1)
	generator = torch.Generator().manual_seed(71)
	u, v = torch.randint(63, (2, 512), generator=generator)
	graph = dgl.graph((u, v), num_nodes=64)
	features = torch.randn(64, 4, generator=generator).to(device)
	model = ModelSpec("sage", 4, 5, 3).build().to(device)
	model.layers[0].activation = torch.sigmoid
	model.layers[0].norm = torch.nn.functional.normalize
	task = AffectedPlanner(graph, 2).block(1, torch.arange(64))
	task = replace(task, block=task.block.to(device))
	engine = IncrementalLayers(model, max_bytes=1024**2)
	for step in range(2):
		if step:
			features[3] += 0.2
		actual = engine.apply(task, features[task.sources])
		expected = model(0, graph.to(device), features)
		torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
	assert len(engine.rows) == 64 and engine.reused > 0
	assert 0 < engine.bytes_used <= engine.max_bytes
