from collections import Counter

import dgl
import pytest
import torch

from lib.graph.methods import neighborhood_policy
from lib.storage.topology import ShardedTopology

pytestmark = pytest.mark.host


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("fanout", [1, 5, 10])
def test_storage_sampling_is_repeatable_preserves_parallel_edges_and_observes_updates(tmp_path, partitions, fanout):
	u = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	v = torch.tensor([11] * 10 + [0, 0])
	graph = dgl.graph((u, v), num_nodes=12)
	targets = torch.tensor([11, 0, 5])
	with ShardedTopology(graph, partitions, directory=tmp_path) as topology:
		first = topology.sample_block(targets, fanout=fanout, seed=71)
		second = topology.sample_block(targets, fanout=fanout, seed=71)
		assert first.in_degrees().tolist() == [min(fanout, 10), min(fanout, 2), 0]
		for a, b in zip(first.edges(), second.edges(), strict=True):
			assert torch.equal(a, b)
		assert torch.equal(first.srcdata[dgl.NID], second.srcdata[dgl.NID])
		starts, ends = first.edges()
		actual = Counter(zip(first.srcdata[dgl.NID][starts].tolist(), targets[ends].tolist(), strict=True))
		original = Counter(zip(u.tolist(), v.tolist(), strict=True))
		assert all(count <= original[pair] for pair, count in actual.items())
		topology.delete_pairs(torch.tensor([9]), torch.tensor([0]))
		topology.add_edges(torch.tensor([4]), torch.tensor([5]))
		block = neighborhood_policy("RTEC-NS", fanout=fanout).block(topology, targets, 1, topology.revision)
		assert block.in_degrees().tolist() == [min(fanout, 10), 1, 1]
		topology.delete_pairs(torch.tensor([0]), torch.tensor([11]))
		topology.add_edges(torch.tensor([0, 0, 7]), torch.tensor([11, 11, 11]))
		first = topology.sample_block(targets, fanout=fanout, seed=71)
		second = topology.sample_block(targets, fanout=fanout, seed=71)
		assert first.in_degrees().tolist() == [min(fanout, 11), 1, 1]
		assert torch.equal(first.srcdata[dgl.NID], second.srcdata[dgl.NID])
		for a, b in zip(first.edges(), second.edges(), strict=True):
			assert torch.equal(a, b)
		starts, ends = first.edges()
		actual = Counter(zip(first.srcdata[dgl.NID][starts].tolist(), targets[ends].tolist(), strict=True))
		original[0, 11] = 2
		original[7, 11] += 1
		original[9, 0] = 0
		original[4, 5] += 1
		assert all(count <= original[pair] for pair, count in actual.items())
		assert not any(part["torch_loaded"] for part in topology.storage.stats().values())


def test_sampling_does_not_transfer_an_entire_high_degree_row(tmp_path):
	graph = dgl.graph((torch.arange(100), torch.full((100,), 100)), num_nodes=101)
	with ShardedTopology(graph, 1, directory=tmp_path, response_bytes=64) as topology:
		with pytest.raises(MemoryError, match="response budget"):
			topology.compile_block(torch.tensor([100]))
		block = topology.sample_block(torch.tensor([100]), fanout=5, seed=71)
		assert block.num_edges() == 5
