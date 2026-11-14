import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from tests.runtime.test_retained_service import _check_retained

pytestmark = pytest.mark.host


@pytest.mark.parametrize("coalesced", [False, True])
@pytest.mark.parametrize("partitions", [1, 3])
def test_retained_sums_with_real_workers_storage_partitions_and_profiling(tmp_path, coalesced, partitions):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 4, 3)
	switches = RuntimeFeatures(
		retain_aggregates=True,
		pipeline=True,
		coalescing=coalesced,
		gather_inputs=True,
		profiling=True,
		distributed_storage=partitions > 1,
	)
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=switches,
		storage_partitions=partitions,
		storage_directory=str(tmp_path),
		worker_bytes=65536,
		storage_bytes=65536,
		batch_size=2,
		timeout=30,
	) as runtime:
		service = runtime.service
		_check_retained(service, spec.build(), graph, features)
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			service.change_features(torch.tensor([0, 3]), torch.full((2, 4), 0.5))
			report = profiler.stop()
		assert report["valid"] and report["traffic_balanced"]
		assert report["engine"]["updates"] == 1
		assert report["coordinator"]["timings"]["root.publish_aggregate"]["count"] > 0
		assert report["coordinator"]["timings"]["root.task_dispatch_and_ready"]["count"] > 0
		assert (
			sum(
				value
				for worker in report["workers"]
				for name, value in worker["counters"].items()
				if name.endswith(".sum_bytes")
			)
			> 0
		)
		_check_retained(service, spec.build(), graph, features)
		u, v = torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5])
		service.insert_edges(u, v)
		if partitions > 1:
			graph.add_edges(u, v)
		_check_retained(service, spec.build(), graph, features)
		service.delete_edges(u, v)
		if partitions > 1:
			_, _, ids = graph.edge_ids(u, v, return_uv=True)
			graph.remove_edges(ids.unique())
		_check_retained(service, spec.build(), graph, features)
		assert service.counters.updates == 3
		assert features.data_ptr() == service.features.data_ptr() and not torch.cuda.is_initialized()
