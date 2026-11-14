from contextlib import ExitStack, closing

import pytest
import torch

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.graph.deferred import DeferredUpdates
from lib.runtime.data_plane import RemoteEmbeddingPlane
from lib.runtime.service import DistributedEmbeddingService
from lib.storage.topology import ShardedTopology
from tests.runtime.test_prepared import _actors
from tests.storage.test_topology import _graph

pytestmark = pytest.mark.host


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("concurrent", [False, True])
def test_storage_processes_drive_exact_and_bounded_real_model_updates(tmp_path, name, partitions, concurrent):
	with ExitStack() as stack:
		graph = _graph()
		topology = stack.enter_context(ShardedTopology(graph, partitions, directory=tmp_path))
		transport, workers = stack.enter_context(_actors(name, gather=True))
		plane = RemoteEmbeddingPlane(transport, storage_bytes=65536)
		stack.callback(plane.close)
		features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
		pointer = features.data_ptr()
		service = DistributedEmbeddingService(
			topology,
			features,
			(4, 4, 3),
			plane,
			batch_size=2,
			frontier_expander=topology.expand,
			source_degree_sensitive=name == "gcn",
			pipeline=True,
			concurrent_compute=concurrent,
			gather_inputs=True,
		)
		deferred = DeferredUpdates(service, max_versions=2, idle_seconds=60)
		stack.callback(deferred.close)
		nodes = torch.tensor([4, 0, 5, 1, 1, 3])
		service.ensure(nodes)
		with closing(UpdateAdmission(service)) as admission:
			for kind, first, second in (
				(UpdateKind.FEATURES, torch.tensor([0, 3]), features[[0, 3]] + 0.25),
				(UpdateKind.INSERT, torch.tensor([0, 5, 5]), torch.tensor([1, 4, 5])),
				(UpdateKind.DELETE, torch.tensor([0, 0, 4]), torch.tensor([1, 1, 0])),
			):
				admission.sync(kind, first, second)
				if kind == UpdateKind.INSERT:
					graph.add_edges(first, second)
				elif kind == UpdateKind.DELETE:
					present = graph.has_edges_between(first, second)
					_, _, ids = graph.edge_ids(first[present], second[present], return_uv=True)
					graph.remove_edges(ids.unique())
				with service.query(nodes, plane.storage.owner) as values, torch.inference_mode():
					model = workers[0].model
					expected = model(1, graph, model(0, graph, features))
					torch.testing.assert_close(values, expected[nodes], rtol=2e-5, atol=2e-5)
			first = admission.submit(UpdateKind.FEATURES, torch.tensor([0, 3]), torch.ones(2, 4))
			second = admission.submit(UpdateKind.FEATURES, torch.tensor([0, 3]), torch.full((2, 4), 2.0))
			admission.sync(UpdateKind.FEATURES, torch.tensor([0, 3]), torch.full((2, 4), 3.0))
			assert first.result(10).input_version == 4 and second.result(10).input_version == 5
			with service.query(nodes, plane.storage.owner) as values, torch.inference_mode():
				expected = model(1, graph, model(0, graph, features))
				torch.testing.assert_close(values, expected[nodes], rtol=2e-5, atol=2e-5)
		assert topology.revision == 2 and topology.num_edges() == graph.num_edges()
		assert service.counters.updates == 6 and service.counters.input_changes == 14
		assert pointer == features.data_ptr() and service.graph is topology
		assert not torch.cuda.is_initialized()
