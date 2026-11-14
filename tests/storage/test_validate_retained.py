from contextlib import ExitStack

import dgl
import torch

from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from scripts.validate.storage import check_sum_rows
from tests.runtime.test_prepared import _actors


def test_selected_sum_validator_covers_duplicate_query_ids_and_isolated_nodes():
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	with ExitStack() as stack:
		transport, workers = stack.enter_context(_actors())
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		service = DistributedEmbeddingService(graph, features, (4, 4, 3), plane, retain_aggregates=True, batch_size=2)
		errors = check_sum_rows(service, workers[0].model, graph, features, torch.tensor([4, 1, 0, 1]))
		assert len(errors) == 2 and max(errors) < 2e-6
