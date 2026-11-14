"""Real CPU processes execute sparse updates and expose routed-work metrics."""

from contextlib import ExitStack

import dgl
import pytest
import torch

from lib.graph.delta_plan import compile_delta_routes
from lib.graph.ownership import StateKey
from lib.models.spec import ModelSpec
from lib.profiling.controller import RuntimeProfiler
from lib.runtime.aggregates.delta import pack_delta_compute
from lib.runtime.aggregates.tables import InputTable, supply_tables
from lib.runtime.config import RuntimeFeatures
from lib.runtime.managed import ManagedRuntime
from lib.runtime.protocol import InputChunk

pytestmark = pytest.mark.host


@pytest.mark.parametrize("coalesced", [False, True])
@torch.inference_mode()
def test_cpu_delta_process_uses_real_peer_sums_and_profiles_routed_edges_without_a_block(coalesced):
	graph = dgl.graph(([0, 0, 1, 2, 3], [1, 1, 2, 0, 3]), num_nodes=6)
	features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
	new_features = features.clone()
	new_features[[0, 3]] += 0.5
	new_graph = graph.clone()
	_, _, ids = new_graph.edge_ids(torch.tensor([0]), torch.tensor([1]), return_uv=True)
	new_graph.remove_edges(ids)
	new_graph.add_edges(torch.tensor([3, 3]), torch.tensor([2, 2]))
	plan = compile_delta_routes(
		new_graph, torch.tensor([0, 3]), torch.tensor([0, 3]), torch.tensor([1, 2]), torch.tensor([-2, 2]), max_edges=64
	)
	spec = ModelSpec("sage", 4, 5, 3)
	model = spec.build()
	with ManagedRuntime(
		graph,
		features,
		spec,
		["cpu", "cpu"],
		switches=RuntimeFeatures(retain_aggregates=True, coalescing=coalesced, gather_inputs=True, profiling=True),
		worker_bytes=65536,
		storage_bytes=65536,
		batch_size=2,
		timeout=30,
	) as runtime:
		service, plane = runtime.service, runtime.plane
		service.ensure(torch.arange(6), 1)
		target = service.workers[1].owner
		old_sum = InputTable(
			plan.targets.numel(),
			5,
			chunks=tuple(
				InputChunk(plane.directory.locate(group.key, preferred=target), group.rows, group.positions)
				for group in service.aggregate_catalog.selections(1, plan.targets, 0)
			),
		)
		tables = (
			old_sum,
			InputTable(plan.targets.numel(), 4, values=new_features[plan.targets]),
			InputTable(plan.inputs.value_nodes.numel(), 4, values=new_features[plan.inputs.value_nodes]),
			InputTable(plan.inputs.value_nodes.numel(), 4, values=features[plan.inputs.value_nodes]),
			InputTable(plan.edges.value_nodes.numel(), 4, values=features[plan.edges.value_nodes]),
		)
		h = plane.directory.begin(StateKey(1, 1, 1000), target)
		a = plane.directory.begin(StateKey(1, 1, 2000, "a"), target)
		packet = pack_delta_compute(plan, h, a, spec.widths, tables, graph.in_degrees(plan.targets))
		with RuntimeProfiler(runtime) as profiler:
			profiler.start()
			with ExitStack() as inputs:
				results = plane.transport.compute_delta(
					target,
					packet,
					output_width=5,
					supply_inputs=lambda: supply_tables(plane, tables, target, inputs, gather=True),
				)
			plane.publish_result(h, results[0])
			plane.publish_result(a, results[1])
			with ExitStack() as leases:
				torch.testing.assert_close(
					plane.read_rows(h.key, torch.arange(plan.targets.numel()), leases),
					model(0, new_graph, new_features)[plan.targets],
					atol=3e-6,
					rtol=3e-6,
				)
			report = profiler.stop()
		assert report["valid"] and report["traffic_balanced"]
		assert report["coordinator"]["timings"]["root.delta_dispatch_and_ready"]["count"] == 1
		worker = next(worker for worker in report["workers"] if worker["rank"] == target.rank)
		assert worker["counters"]["delta.layer1.executions"] == 1
		assert (
			worker["counters"]["delta.layer1.routed_edges"] == plan.inputs.sources.numel() + plan.edges.sources.numel()
		)
		assert not any(name.endswith("compiled_edges") for name in worker["counters"])
		assert worker["timings"]["worker.delta_cpu.layer1"]["count"] == 1
		assert not torch.cuda.is_initialized()
