import dgl
import torch

from lib.storage.pma import PMATopology


def test_pma_growth_deletion_parallel_edges_and_block_planning():
	torch.set_num_threads(1)
	graph = dgl.graph(([], []), num_nodes=9)
	pma = PMATopology(graph)
	generator = torch.Generator().manual_seed(7)
	for step in range(80):
		u, v = torch.randint(9, (2, 8), generator=generator)
		if step % 3:
			graph.add_edges(u, v)
			pma.add_edges(u, v)
		else:
			pma.delete_pairs(u, v)
			mask = graph.has_edges_between(u, v)
			if mask.any():
				graph.remove_edges(graph.edge_ids(u[mask], v[mask], return_uv=True)[2].unique())
		assert pma.num_edges() == graph.num_edges()
		assert torch.equal(pma.in_degrees(), graph.in_degrees())
		assert torch.equal(pma.out_degrees(), graph.out_degrees())
		for node in range(9):
			assert torch.equal(
				pma.out_edges(torch.tensor([node]))[1].sort().values, graph.out_edges(node)[1].sort().values
			)
		targets = torch.tensor([7, 0, 2])
		block = pma.compile_block(targets)
		bu, bv = block.edges()
		pairs = torch.stack([block.srcdata[dgl.NID][bu], targets[bv]], 1)
		gu, gv = graph.in_edges(targets)
		assert sorted(map(tuple, pairs.tolist())) == sorted(zip(gu.tolist(), gv.tolist(), strict=True))
