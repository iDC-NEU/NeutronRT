import dgl
import torch

from lib.models.spec import ModelSpec
from scripts.evaluate.aggregate_storage import evaluate


def test_comparison_covers_all_selected_rows_and_charges_degrees_and_embeddings():
	graph = dgl.graph(([0, 0, 1, 2], [1, 1, 2, 0]), num_nodes=5)
	features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
	model = ModelSpec("sage", 4, 3, 2).build()
	queries = torch.tensor([4, 2, 0])
	labels = torch.tensor([1, 0, 0, 1, 1])
	report = evaluate(graph, features, model, queries, labels, repetitions=2, query_rows=2, chunk_edges=1)
	assert report["features_bytes"] == 5 * 4 * 4
	assert report["current_embedding_only_bytes"] == 5 * (3 + 2) * 4
	for setting, width in zip(report["settings"], (4 + 3, 3 + 2), strict=True):
		assert setting["valid"] and setting["aggregate_only_bytes"] == 5 * (width * 4 + 2 * 8)
		assert setting["saved_embedding_bytes"] == report["current_embedding_only_bytes"]
		assert setting["accuracy"]["rows"] == 3 and setting["accuracy"]["prediction_disagreement"] == 0
		assert len(setting["raw_cached_read_seconds"]) == len(setting["raw_reconstructed_read_seconds"]) == 2
