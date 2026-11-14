import dgl
import pytest
import torch

from lib.graph.delta_capture import capture_edges
from lib.storage.topology import ShardedTopology

pytestmark = pytest.mark.host


@pytest.mark.parametrize("partitions", [1, 3])
def test_signed_edge_capture_uses_current_storage_multiplicities_before_mutation(tmp_path, partitions):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	with ShardedTopology(graph, partitions, directory=tmp_path) as topology:
		for delete, u, v in (
			(False, torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5])),
			(True, torch.tensor([0, 0, 4, 5]), torch.tensor([1, 1, 2, 5])),
			(True, torch.tensor([0, 4]), torch.tensor([1, 2])),
		):
			revision = topology.revision
			actual = capture_edges(topology, u, v, delete=delete)
			expected = capture_edges(graph, u, v, delete=delete)
			assert topology.revision == revision
			for first, second in zip(
				(actual.sources, actual.targets, actual.weights),
				(expected.sources, expected.targets, expected.weights),
				strict=True,
			):
				torch.testing.assert_close(first, second)
			before = topology.num_edges()
			if delete:
				topology.delete_pairs(u, v)
				present = graph.has_edges_between(u, v)
				if present.any():
					_, _, ids = graph.edge_ids(u[present], v[present], return_uv=True)
					graph.remove_edges(ids.unique())
			else:
				topology.add_edges(u, v)
				graph.add_edges(u, v)
			assert topology.num_edges() - before == int(actual.weights.sum())
			assert topology.num_edges() == graph.num_edges()
		assert all(not item["response_buffers"] for item in topology.storage.stats().values())
