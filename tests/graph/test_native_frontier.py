import dgl
import pytest
import torch

from lib.graph.affected import AffectedPlanner
from lib.graph.native import compile_native
from lib.graph.native_frontier import NativeFrontier


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
	return compile_native(tmp_path_factory.mktemp("native-frontier"))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("degree_sensitive", [False, True])
def test_native_affected_analysis_borrows_csr_and_matches_python_after_topology_changes(
	native_library, threads, degree_sensitive
):
	graph = dgl.graph(([0, 0, 1, 2, 2, 3, 4], [1, 1, 2, 3, 4, 2, 5]), num_nodes=8)
	native = NativeFrontier(graph, threads=threads, library=native_library)
	assert native.offsets.data_ptr() == graph.adj_tensors("csr")[0].data_ptr()
	assert native.neighbors.data_ptr() == graph.adj_tensors("csr")[1].data_ptr()
	baseline = AffectedPlanner(graph, 3, source_degree_sensitive=degree_sensitive)
	optimized = AffectedPlanner(graph, 3, source_degree_sensitive=degree_sensitive, expander=native)
	for mutation in ("original", "insert", "delete"):
		if mutation == "insert":
			graph.add_edges(torch.tensor([1, 6]), torch.tensor([6, 7]))
		elif mutation == "delete":
			graph.remove_edges(torch.tensor([0, 2, 5]))
		for features, edges, degrees in (([0], [], []), ([], [7], []), ([4], [6], [1]), ([], [], [])):
			arguments = {
				"changed_features": torch.tensor(features, dtype=torch.int64),
				"edge_targets": torch.tensor(edges, dtype=torch.int64),
				"degree_sources": torch.tensor(degrees, dtype=torch.int64),
			}
			for expected, actual in zip(baseline.analyze(**arguments), optimized.analyze(**arguments), strict=True):
				assert expected.layer == actual.layer
				assert torch.equal(expected.targets, actual.targets)
		assert native.offsets.data_ptr() == graph.adj_tensors("csr")[0].data_ptr()
		assert native.neighbors.data_ptr() == graph.adj_tensors("csr")[1].data_ptr()


def test_native_rejects_invalid_inputs_before_entering_c_abi(native_library):
	graph = dgl.graph(([0], [1]), num_nodes=3)
	native = NativeFrontier(graph, library=native_library)
	for nodes in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.0]), torch.ones(2, 1, dtype=torch.int64)):
		with pytest.raises(ValueError):
			native(nodes, torch.empty(0, dtype=torch.int64))
	with pytest.raises(ValueError, match="thread"):
		NativeFrontier(graph, threads=False, library=native_library)
	assert native(torch.tensor([0, 0]), torch.tensor([2, 2])).tolist() == [0, 1, 2]
