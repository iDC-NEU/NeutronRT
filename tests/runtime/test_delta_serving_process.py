"""Public sparse serving across independent local storage/compute processes."""

from contextlib import ExitStack, closing

import dgl
import pytest
import torch

from lib.graph.admission import UpdateAdmission, UpdateKind
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime

pytestmark = pytest.mark.host


@pytest.mark.parametrize("partitions,coalesced,peer", [(1, False, True), (3, True, False)])
@pytest.mark.parametrize("row_refresh", [False, True])
@pytest.mark.parametrize("aggregate_only", [False, True])
@torch.no_grad()
def test_sparse_serving_with_three_layers_tight_memory_async_and_refresh(
	partitions, coalesced, peer, row_refresh, aggregate_only, tmp_path
):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	reference = graph.clone()
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	pointer = features.data_ptr()
	spec = ModelSpec("sage", 4, 5, 3, layers=3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=RuntimeFeatures(
			pipeline=True,
			coalescing=coalesced,
			peer_fetch=peer,
			gather_inputs=True,
			profiling=True,
			retain_aggregates=True,
			aggregate_only=aggregate_only,
			incremental_aggregates=True,
			row_refresh=row_refresh,
			distributed_storage=partitions > 1,
		),
		storage_partitions=partitions,
		storage_directory=str(tmp_path),
		worker_bytes=512,
		storage_bytes=65536,
		batch_size=2,
		delta_max_steps=2,
		timeout=30,
	) as runtime:
		service, nodes = runtime.service, torch.arange(6)
		service.ensure(nodes)

		def check():
			current = features
			for index in range(3):
				projected = model.layers[index].fc_neigh(current)
				sums = torch.zeros_like(projected)
				u, v = reference.edges()
				sums.index_add_(0, v, projected[u])
				current = model(index, reference, current)
				with service.query(nodes, runtime.plane.storage.owner, layer=index + 1) as actual:
					torch.testing.assert_close(actual, current, rtol=1e-5, atol=1e-5)
				for group in service.aggregate_catalog.selections(index + 1, nodes, service.catalog.version):
					with ExitStack() as leases:
						actual = runtime.plane.read_rows(group.key, group.rows, leases)
						torch.testing.assert_close(actual, sums[group.positions], rtol=1e-5, atol=1e-5)
			assert service._delta.state.active is None and not runtime.plane._retained_states

		check()
		with RuntimeProfiler(runtime) as profiler, closing(UpdateAdmission(service)) as admission:
			profiler.start()
			admission.sync(UpdateKind.FEATURES, torch.tensor([0, 3]), torch.ones(2, 4))
			check()
			u, v = torch.tensor([0, 0, 5]), torch.tensor([1, 1, 5])
			admission.sync(UpdateKind.INSERT, u, v)
			reference.add_edges(u, v)
			check()
			admission.sync(UpdateKind.DELETE, u, v)
			_, _, identifiers = reference.edge_ids(u, v, return_uv=True)
			reference.remove_edges(identifiers.unique())
			check()
			tickets = [
				admission.submit(UpdateKind.FEATURES, torch.tensor([0, 3]), torch.full((2, 4), index / 5))
				for index in range(4)
			]
			assert [ticket.result(30).version for ticket in tickets] == [4, 5, 6, 7]
			check()
			report = profiler.stop()
		assert report["valid"] and report["traffic_balanced"]
		assert service.counters.delta_tasks > 0 and service.counters.delta_refresh_rows > 0
		assert service._delta.state.row_refresh == row_refresh
		if aggregate_only:
			assert service.counters.reconstructed_rows > 0
			assert all(key.kind == "a" or service.catalog.current_nodes(key).numel() for key in runtime.plane._shapes)
			assert all(
				key.kind == "a"
				for key in runtime.plane._shapes
				if runtime.plane.directory.locate(key, preferred=runtime.plane.storage.owner).owner
				== runtime.plane.storage.owner
			)
		assert service._delta.state.depth(list(service._delta.state._depths)) <= 2
		assert report["delta_state"]["row_refresh"] == row_refresh
		assert (report["delta_state"]["depth_tensor_bytes"] > 0) == row_refresh
		assert (
			sum(
				value
				for worker in report["workers"]
				for name, value in worker["counters"].items()
				if name.startswith("delta.") and name.endswith("executions")
			)
			== service.counters.delta_tasks
		)
		assert features.data_ptr() == pointer and not torch.cuda.is_initialized()
