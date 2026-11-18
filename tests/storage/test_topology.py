import os
from concurrent.futures import ThreadPoolExecutor

import dgl
import numpy as np
import pytest
import torch

from lib.graph.delta_plan import compile_delta_routes
from lib.graph.deltas import update_neighbor_sums
from lib.graph.recompute import LayerPlan
from lib.models.operators import full_neighbor_forward
from lib.models.spec import ModelSpec
from lib.storage.base import BaseTopology
from lib.storage.topology import ShardedTopology

pytestmark = pytest.mark.host


def _graph():
	return dgl.graph(([0, 0, 0, 1, 1, 2, 3], [0, 1, 1, 2, 3, 3, 4]), num_nodes=6)


def _check(topology, graph):
	nodes = torch.tensor([4, 0, 5, 1, 1, 3])
	torch.testing.assert_close(topology.in_degrees(nodes), graph.in_degrees(nodes))
	torch.testing.assert_close(topology.out_degrees(nodes), graph.out_degrees(nodes))
	u, v = torch.tensor([0, 0, 4, 5]), torch.tensor([1, 5, 3, 5])
	torch.testing.assert_close(topology.has_edges_between(u, v), graph.has_edges_between(u, v))
	gu, gv = graph.edges()
	counts = torch.tensor([int(((gu == source) & (gv == target)).sum()) for source, target in zip(u, v, strict=True)])
	torch.testing.assert_close(topology.edge_multiplicities(u, v), counts)
	frontier, edges = torch.tensor([3, 0]), torch.tensor([5, 2])
	expected = torch.cat([frontier, graph.out_edges(frontier)[1], edges]).unique(sorted=True)
	torch.testing.assert_close(topology.expand(frontier, edges), expected)
	first, second = topology.out_edges(frontier)
	u, v = graph.out_edges(frontier)
	torch.testing.assert_close(
		(first * graph.num_nodes() + second).sort().values, (u * graph.num_nodes() + v).sort().values
	)
	assert topology.num_edges() == graph.num_edges()
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	empty = torch.empty(0, dtype=torch.int64)
	plan = compile_delta_routes(topology, frontier, empty, empty, empty, max_edges=64)
	old_sums = torch.zeros_like(features)
	u, v = graph.edges()
	old_sums.index_add_(0, v, features[u])
	changed_features = features.clone()
	changed_features[frontier] += 0.25
	expected_sums = torch.zeros_like(features)
	expected_sums.index_add_(0, v, changed_features[u])
	actual = update_neighbor_sums(
		old_sums[plan.targets],
		graph.in_degrees(plan.targets),
		plan.inputs.bind((changed_features - features)[plan.inputs.value_nodes]),
		plan.edges.bind(features[plan.edges.value_nodes]),
	)
	torch.testing.assert_close(actual.sums, expected_sums[plan.targets])
	targets = torch.tensor([5, 4, 0, 1, 3])
	block = topology.compile_block(targets)
	for name in ("sage", "gat", "gcn"):
		model = ModelSpec(name, 4, 4, 3).build()
		with torch.inference_mode():
			if name == "gcn":
				# Serving's source-degree normalization uses the full topology.
				actual = full_neighbor_forward(
					model.layers[0],
					LayerPlan(1, block, block.srcdata[dgl.NID], targets, topology.out_degrees(block.srcdata[dgl.NID])),
					features[block.srcdata[dgl.NID]],
				).relu()
			else:
				actual = model(0, block, features[block.srcdata[dgl.NID]])
			torch.testing.assert_close(actual, model(0, graph, features)[targets], rtol=2e-5, atol=2e-5)
	assert all(
		not item["torch_loaded"] and not item["cuda_initialized"] and not item["response_buffers"]
		for item in topology.storage.stats().values()
	)


@pytest.mark.parametrize("partitions", [1, 2, 3, 8])
def test_process_topology_preserves_rows_blocks_updates_and_response_lifetimes(tmp_path, partitions):
	graph = _graph()
	with ShardedTopology(graph, partitions, directory=tmp_path) as topology:
		processes = list(topology.storage.processes)
		folder = topology.base.folder
		names = []
		original = topology.storage.exchange

		def record(requests, **kwargs):
			result = original(requests, **kwargs)
			for index, request in requests.items():
				if request[0] == "rows":
					fd = result[index][0]
					names.append((fd, os.fstat(fd).st_ino))
			return result

		topology.storage.exchange = record
		_check(topology, graph)
		for delete, sources, targets in (
			(False, torch.tensor([0, 5, 5]), torch.tensor([1, 4, 5])),
			(True, torch.tensor([0, 0, 4]), torch.tensor([1, 1, 0])),
			(False, torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
		):
			if delete:
				topology.delete_pairs(sources, targets)
				present = graph.has_edges_between(sources, targets)
				_, _, ids = graph.edge_ids(sources[present], targets[present], return_uv=True)
				graph.remove_edges(ids.unique())
			else:
				topology.add_edges(sources, targets)
				graph.add_edges(sources, targets)
			_check(topology, graph)
		assert topology.revision == 3
		assert all(item["revision"] == 3 for item in topology.storage.stats().values())
		with ThreadPoolExecutor(2) as executor:
			list(executor.map(lambda _: _check(topology, graph.local_var()), range(4)))
		for fd, inode in names:
			try:
				current = os.fstat(fd)
			except OSError:
				continue
			# Other threads may reuse an FD number after our mapping closes.
			assert current.st_ino != inode
	assert not folder.exists() and all(not process.is_alive() for process in processes)


def test_budget_failure_aborts_both_indexes_on_every_partition(tmp_path):
	graph = _graph()
	with ShardedTopology(graph, 3, directory=tmp_path, overlay_bytes=16, response_bytes=1024) as topology:
		with pytest.raises(MemoryError, match="overlay budget"):
			topology.add_edges(torch.tensor([0, 5]), torch.tensor([5, 0]))
		assert topology.revision == 0 and topology.num_edges() == graph.num_edges()
		assert all(item["overlay_rows"] == 0 for item in topology.storage.stats().values())
		_check(topology, graph)
		topology.add_edges(torch.tensor([5]), torch.tensor([5]))
		assert topology.revision == 1 and topology.num_edges() == graph.num_edges() + 1


def test_failed_shared_response_releases_other_partition_buffers(tmp_path):
	with ShardedTopology(_graph(), 3, directory=tmp_path, response_bytes=24) as topology:
		with pytest.raises(MemoryError, match="response budget"):
			topology.compile_block(torch.tensor([0, 1, 4, 5]))
		assert all(item["response_buffers"] == 0 for item in topology.storage.stats().values())
		block = topology.compile_block(torch.tensor([4, 5]))
		assert block.num_edges() == 1


@pytest.mark.parametrize("partitions", [1, 3])
def test_expansion_spans_bounded_responses_and_preserves_updated_reachability(tmp_path, partitions):
	graph = _graph()
	with ShardedTopology(graph, partitions, directory=tmp_path, response_bytes=32) as topology:
		frontier, extra = torch.tensor([0, 3, 1, 4, 0]), torch.tensor([5, 2])
		for insert in (False, True):
			if insert:
				topology.add_edges(torch.tensor([5]), torch.tensor([0]))
				graph.add_edges(torch.tensor([5]), torch.tensor([0]))
				frontier = torch.cat((frontier, torch.tensor([5])))
			expected = torch.cat((frontier, graph.out_edges(frontier)[1], extra)).unique(sorted=True)
			torch.testing.assert_close(topology.expand(frontier, extra), expected)
			assert all(item["response_buffers"] == 0 for item in topology.storage.stats().values())


def test_expansion_retains_the_single_row_capacity_limit(tmp_path):
	with ShardedTopology(_graph(), 1, directory=tmp_path, response_bytes=16) as topology:
		with pytest.raises(MemoryError, match="single outgoing row"):
			topology.expand(torch.tensor([0]), torch.empty(0, dtype=torch.int64))
		assert all(item["response_buffers"] == 0 for item in topology.storage.stats().values())


def test_export_is_readonly_mapped_and_rejects_nonempty_directories(tmp_path):
	graph = _graph()
	formats = graph.formats()
	base = BaseTopology.export(graph, tmp_path)
	assert graph.formats() == formats
	for direction in ("in", "out"):
		assert all(isinstance(np.load(path, mmap_mode="r"), np.memmap) for path in base.files(direction))
	with pytest.raises(ValueError, match="empty"):
		BaseTopology.export(_graph(), tmp_path)


def test_persistent_exported_base_is_verified_and_survives_topology_close(tmp_path):
	graph = _graph()
	base_folder = tmp_path / "persistent-base"
	BaseTopology.export(graph, base_folder)
	with ShardedTopology(graph, 1, exported_base=base_folder) as topology:
		assert topology.base.folder == base_folder.resolve()
		assert topology.initialization["base_fingerprint"]
	assert base_folder.exists()

	wrong = dgl.graph(([0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 2]), num_nodes=6)
	with pytest.raises(ValueError, match="graph content"):
		ShardedTopology(wrong, 1, exported_base=base_folder)


def test_remote_storage_requires_a_persistent_exported_base(tmp_path):
	with pytest.raises(ValueError, match="pre-exported persistent base"):
		ShardedTopology(_graph(), 1, storage_endpoints=("127.0.0.1:1",))


def test_empty_graph_and_invalid_requests_do_not_change_partition_state(tmp_path):
	empty = torch.empty(0, dtype=torch.int64)
	with ShardedTopology(dgl.graph((empty, empty), num_nodes=0), 3, directory=tmp_path) as topology:
		assert topology.compile_block(empty).num_edges() == 0
		assert topology.expand(empty, empty).numel() == 0
		assert all(not values.numel() for values in topology.out_edges(empty))
		topology.add_edges(empty, empty)
		assert topology.revision == 1 and topology.num_edges() == 0
	with ShardedTopology(_graph(), 2, directory=tmp_path) as topology:
		with pytest.raises(ValueError, match="unique"):
			topology.out_edges(torch.tensor([0, 0]))
		for first, second in (
			(torch.tensor([-1]), torch.tensor([0])),
			(torch.tensor([0]), torch.tensor([6])),
			(torch.tensor([0]), torch.tensor([1, 2])),
			(torch.tensor([0.0]), torch.tensor([1])),
		):
			with pytest.raises(ValueError):
				topology.add_edges(first, second)
		assert topology.revision == 0 and topology.num_edges() == 7
