import shutil

import dgl
import pytest
import torch

from lib.graph.native import NativeDeltaAccumulator, compile_native


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
	if shutil.which("g++") is None:
		pytest.skip("Optional native kernels require g++")
	return compile_native(tmp_path_factory.mktemp("native-delta"))


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_native_threads_update_existing_storage(native_library, threads, strided):
	graph = dgl.graph(([0, 0, 1, 2, 3, 4], [1, 1, 1, 3, 4, 0]), num_nodes=6)
	current = torch.arange(24, dtype=torch.float32).reshape(6, 4)
	if strided:
		current = current.t().contiguous().t()
	changed = torch.tensor([3, 0])
	previous = current[changed] - 2
	targets = torch.tensor([0, 1, 4])
	output = torch.zeros_like(current, dtype=torch.float64, memory_format=torch.contiguous_format)
	expected = output.clone()
	u, v = graph.out_edges(changed)
	mapping = torch.full((6,), -1, dtype=torch.int64)
	mapping[changed] = torch.arange(2)
	expected.index_add_(0, v, current[u].double() - previous[mapping[u]].double())
	native = NativeDeltaAccumulator(graph, native_library)
	pointers = current.data_ptr(), previous.data_ptr(), output.data_ptr(), native.offsets.data_ptr()
	native.apply(current, previous, changed, targets, output, threads=threads)
	assert pointers == (current.data_ptr(), previous.data_ptr(), output.data_ptr(), native.offsets.data_ptr())
	torch.testing.assert_close(output, expected)
	# Reusing the mapping must clear sources left over from the previous call.
	output.zero_()
	native.apply(current, current[[1]], torch.tensor([1]), torch.tensor([1]), output, threads=threads)
	assert not output.any()


def test_native_rejects_stale_graph_views_and_invalid_ids_before_writing(native_library):
	graph = dgl.graph(([0], [1]), num_nodes=3)
	native = NativeDeltaAccumulator(graph, native_library)
	current = torch.ones(3, 2)
	output = torch.zeros(3, 2, dtype=torch.float64)
	for changed, targets in (
		(torch.tensor([0, 0]), torch.tensor([1])),
		(torch.tensor([0]), torch.tensor([4])),
	):
		with pytest.raises(ValueError):
			native.apply(current, current[changed], changed, targets, output)
		assert not output.any()
	graph.add_edges(torch.tensor([2]), torch.tensor([1]))
	with pytest.raises(ValueError, match="topology changed"):
		native.apply(current, current[[0]], torch.tensor([0]), torch.tensor([1]), output)
	assert not output.any()


def test_native_instance_serializes_concurrent_mapping_changes(native_library):
	from concurrent.futures import ThreadPoolExecutor

	graph = dgl.graph(([0, 1, 2], [3, 3, 3]), num_nodes=4)
	native = NativeDeltaAccumulator(graph, native_library)
	current = torch.arange(12, dtype=torch.float32).reshape(4, 3)

	def call(source):
		output = torch.zeros(4, 3, dtype=torch.float64)
		changed = torch.tensor([source])
		for _ in range(20):
			native.apply(current, current[changed] - 1, changed, torch.tensor([3]), output, threads=2)
		return output

	with ThreadPoolExecutor(max_workers=2) as pool:
		results = list(pool.map(call, [0, 2]))
	for result in results:
		torch.testing.assert_close(result[3], torch.full((3,), 20.0, dtype=torch.float64))
		assert not result[:3].any()
