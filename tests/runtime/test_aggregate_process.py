"""Actual CPU worker processes for paired sum/output publication and writeback."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.ownership import StateKey
from lib.graph.recompute import RecomputePlanner
from lib.graph.serving import select_rows
from lib.models.spec import ModelSpec
from lib.runtime.aggregates.protocol import pack_aggregate_compute
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.protocol import InputChunk

pytestmark = pytest.mark.host


@pytest.mark.parametrize("coalesced", [False, True])
@torch.inference_mode()
def test_cpu_process_pairs_publish_read_peer_rows_and_write_back_sums(coalesced):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	spec = ModelSpec("sage", 4, 4, 3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=RuntimeFeatures(coalescing=coalesced),
		worker_bytes=65536,
		storage_bytes=65536,
		timeout=30,
	) as runtime:
		plane = runtime.service.remote_plane
		workers = runtime.service.workers
		plan = RecomputePlanner(graph, 2).plan(torch.arange(6), 2, 0)
		previous = None
		current = features
		keys = []
		for task in plan.tasks:
			worker = workers[task.layer - 1]
			h = plane.directory.begin(StateKey(0, task.layer, 700 + task.layer), worker.owner)
			a = plane.directory.begin(StateKey(0, task.layer, 800 + task.layer, "a"), worker.owner)
			inputs = (
				() if task.layer == 1 else (InputChunk(previous, task.sources, torch.arange(task.sources.numel())),)
			)
			packet = pack_aggregate_compute(
				task,
				h,
				a,
				width=4,
				inputs=inputs,
				features=select_rows(features, task.sources) if task.layer == 1 else None,
			)
			with ExitStack() as leases:

				def supply(inputs=inputs, worker=worker, leases=leases):
					plane.supply_inputs(inputs, worker.owner, 4, leases)

				results = plane.transport.compute_with_aggregate(
					worker.owner,
					packet,
					output_width=4 if task.layer == 1 else 3,
					supply_inputs=supply,
				)
			previous = plane.publish_result(h, results[0])
			plane.publish_result(a, results[1])
			keys.extend((h.key, a.key))
			projected = model.layers[task.layer - 1].fc_neigh(current)
			expected_sums = torch.zeros_like(projected)
			u, v = graph.edges()
			expected_sums.index_add_(0, v, projected[u])
			current = model(task.layer - 1, graph, current)
			rows = torch.tensor([5, 1, 0, 1])
			with ExitStack() as leases:
				torch.testing.assert_close(plane.read_rows(h.key, rows, leases), current[rows], atol=2e-6, rtol=2e-6)
				torch.testing.assert_close(plane.read_rows(a.key, rows, leases), expected_sums[rows])
			plane.writeback(a.key, plane.storage.owner).result(15)
			plane.evict(a.key, worker.owner)
			with ExitStack() as leases:
				torch.testing.assert_close(plane.read_rows(a.key, rows, leases), expected_sums[rows])
		for key in keys:
			plane.retire(key)
		assert features.data_ptr() == runtime.service.features.data_ptr()
		assert not torch.cuda.is_initialized()
