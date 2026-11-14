"""Cross-process reconstruction reuses the live aggregate catalogs and profiler."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.ownership import StateKey
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.aggregates.reconstruction import pack_reconstruction
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.protocol import InputChunk

pytestmark = pytest.mark.host


@pytest.mark.parametrize("coalesced", [False, True])
@torch.inference_mode()
def test_cpu_process_reconstruction_from_live_sums_counts_transforms_without_compiled_edges(tmp_path, coalesced):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 5, 3)
	model = spec.build()
	expected = model(1, graph, model(0, graph, features))
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=RuntimeFeatures(
			retain_aggregates=True, coalescing=coalesced, gather_inputs=True, profiling=True, distributed_storage=True
		),
		storage_partitions=3,
		storage_directory=str(tmp_path),
		worker_bytes=65536,
		storage_bytes=65536,
		batch_size=2,
		timeout=30,
	) as runtime:
		service, plane = runtime.service, runtime.plane
		service.ensure(torch.arange(6))
		rows = torch.tensor([5, 1, 0, 1])
		target = service.workers[1].owner
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			for start in (0, 1):
				tables = [InputTable(rows.numel(), 4, values=features[rows])] if not start else []
				for layer in range(max(1, start), 3):
					catalog = service.catalog if layer == start else service.aggregate_catalog
					chunks = tuple(
						InputChunk(plane.directory.locate(group.key, preferred=target), group.rows, group.positions)
						for group in catalog.selections(layer, rows, 0)
					)
					tables.append(InputTable(rows.numel(), spec.widths[layer], chunks=chunks))
				attempt = plane.directory.begin(StateKey(0, 2, 1000 + start), target)
				packet = pack_reconstruction(
					attempt, spec.widths, tables, service.graph.in_degrees(rows), start_layer=start
				)
				with ExitStack() as leases:
					result = plane.transport.reconstruct(
						target,
						packet,
						output_width=3,
						supply_inputs=lambda tables=tables, leases=leases: supply_tables(
							plane, tables, target, leases, gather=True
						),
					)
				plane.publish_result(attempt, result)
				with ExitStack() as leases:
					torch.testing.assert_close(
						plane.read_rows(attempt.key, torch.arange(rows.numel()), leases),
						expected[rows],
						atol=2e-6,
						rtol=2e-6,
					)
				plane.retire(attempt.key)
			report = profiler.stop()
		assert report["valid"] and report["traffic_balanced"]
		assert report["coordinator"]["timings"]["root.reconstruction_dispatch_and_ready"]["count"] == 2
		worker = next(worker for worker in report["workers"] if worker["rank"] == target.rank)
		assert worker["counters"]["reconstruction.layer2.rows"] == 8
		assert worker["counters"]["reconstruction.transforms"] == 3
		assert worker["timings"]["worker.reconstruction_cpu.layer2"]["count"] == 2
		assert not any(name.endswith("compiled_edges") for name in worker["counters"])
		assert not torch.cuda.is_initialized()
