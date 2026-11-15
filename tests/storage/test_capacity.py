import dgl
import pytest

from lib.storage.base import BaseTopology
from lib.storage.capacity import GIB, StorageNodeBudget, manifest, require_capacity


def test_capacity_manifest_splits_rows_and_reserves_memory(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	base = BaseTopology.export(graph, tmp_path)
	result = manifest(
		base,
		[
			StorageNodeBudget("node-a", capacity_bytes=1 * GIB, reserve_bytes=128),
			StorageNodeBudget("node-b", capacity_bytes=1 * GIB, reserve_bytes=128),
		],
	)
	assert [item["start"] for item in result["partitions"]] == [0, 2]
	assert all(item["fits"] for item in result["partitions"])
	assert result["nodes"][0]["usable_bytes"] == GIB - 128


def test_capacity_rejects_a_partition_over_usable_budget(tmp_path):
	graph = dgl.graph(([0, 1, 2], [1, 2, 3]), num_nodes=4)
	base = BaseTopology.export(graph, tmp_path)
	with pytest.raises(MemoryError, match="exceeds"):
		require_capacity(base, [StorageNodeBudget("node", capacity_bytes=1, reserve_bytes=0)])
