"""Observe the frozen ICDE kernels without changing their execution logic."""

import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from benchmarks.sources import PinnedSource


def verify(
	checkout: Path, output: Path, *, fanouts: Sequence[int], seed: int = 71, sampling_seeds: int = 8
) -> dict[str, Any]:
	"""Check full recomputation, sampling dependencies and incremental/cache kernels.

	The source methods discard final outputs. The observer records those outputs
	and their input IDs; it does not replace sampling, model math or state writes.
	"""
	if not fanouts or any(value <= 0 for value in fanouts) or sampling_seeds <= 0:
		raise ValueError("Fanouts and sampling seed count must be positive")
	if "_icde_lib" in sys.modules:
		raise RuntimeError("Run ICDE verification in a fresh process")
	output = output.resolve()
	pin = PinnedSource.load("icde")
	output.mkdir(parents=True, exist_ok=False)
	source = pin.snapshot(checkout, output / "source")
	paths = [
		source / name
		for name in (
			"lib/models/__init__.py",
			"lib/models/convs.py",
			"lib/methods/smp.py",
			"lib/methods/cache.py",
			"lib/methods/inc.py",
		)
	]
	hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
	package = ModuleType("_icde_lib")
	package.__path__ = [str(source / "lib")]
	sys.modules[package.__name__] = package
	models = importlib.import_module("_icde_lib.models")
	sampled = importlib.import_module("_icde_lib.methods.smp")
	cached = importlib.import_module("_icde_lib.methods.cache")
	incremental = importlib.import_module("_icde_lib.methods.inc")

	import dgl
	import torch
	import torch.distributed as dist

	from lib.models import GNNModel

	if dist.is_initialized():
		raise RuntimeError("Run ICDE verification without an existing process group")
	torch.set_num_threads(1)
	torch.manual_seed(seed)
	count, features, hidden, classes, layers = 32, 4, 8, 3, 2
	nodes = torch.arange(count)
	graph = dgl.graph((nodes.repeat(9), torch.cat([(nodes + i) % count for i in range(9)])), num_nodes=count)
	x = torch.randn(count, features) * 0.1
	model = models.GNNModel("sage", features, hidden, classes, layers, False, aggregator_type="mean").eval()
	oracle = GNNModel("sage", features, hidden, classes, layers).eval()
	oracle.load_state_dict(model.state_dict(), strict=True)
	with torch.inference_mode():
		initial_first = oracle(0, graph, x)
		initial_aggregates = []
		for values in (x, initial_first):
			neighbor_sum = torch.zeros_like(values)
			source_ids, target_ids = graph.edges()
			neighbor_sum.index_add_(0, target_ids, values[source_ids])
			initial_aggregates.append(neighbor_sum / graph.in_degrees().clamp_min(1).unsqueeze(1))
	graph.add_edges(0, 16)
	with torch.inference_mode():
		expected_first = oracle(0, graph, x)
		expected = oracle(1, graph, expected_first)
	updates = [torch.tensor([graph.num_edges() - 1])]
	fixture_hash = hashlib.sha256()
	for value in (*graph.edges(), x, *model.state_dict().values()):
		fixture_hash.update(value.contiguous().numpy().tobytes())

	class Observer(torch.nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.module = model
			self.device = torch.device("cpu")
			self.outputs = torch.full_like(expected, float("nan"))
			self.computed: set[int] = set()
			self.missing: set[int] = set()

		def forward(self, layer, block, h):
			result = self.module(layer, block, h)
			if layer == 0:
				self.computed.update(block.dstdata[dgl.NID].tolist())
			else:
				self.missing.update(set(block.srcdata[dgl.NID].tolist()) - self.computed)
				self.outputs[block.dstdata[dgl.NID]] = result.detach()
			return result

	records = []
	dist.init_process_group("gloo", init_method=(output / "process-group.store").as_uri(), rank=0, world_size=1)
	try:
		for fanout in [-1, *dict.fromkeys(fanouts)]:
			for offset in range(1 if fanout == -1 else sampling_seeds):
				sampling_seed = seed + offset
				dgl.seed(sampling_seed)
				torch.manual_seed(sampling_seed)
				method = sampled.Sample(layers)
				samplers = [dgl.dataloading.NeighborSampler([fanout]) for _ in range(layers)]
				observer = Observer()
				method.affect_analysis(graph, updates, samplers)
				with torch.inference_mode():
					method.layerwise_inference(graph, observer, samplers, ([x.clone(), initial_first.clone()], None))
				targets = method.collect_nodes[0][-1]
				actual = observer.outputs[targets]
				exact = fanout == -1 or fanout >= int(graph.in_degrees().max())
				matches_full = bool(torch.allclose(actual, expected[targets], atol=1e-6, rtol=1e-5))
				records.append(
					{
						"implementation": "ICDE-Full" if fanout == -1 else "ICDE-NS",
						"fanout": fanout,
						"sampling_seed": sampling_seed,
						"rows_checked": len(targets),
						"uncomputed_dependencies": sorted(observer.missing),
						"requires_exact_output": exact,
						"matches_full": matches_full,
						"max_error_to_full": float((actual - expected[targets]).abs().max()),
						"valid": bool(torch.isfinite(actual).all())
						and not observer.missing
						and (not exact or matches_full),
					}
				)
		method = cached.Cache(layers)
		samplers = [dgl.dataloading.NeighborSampler([-1]) for _ in range(layers)]
		observer = Observer()
		state = [x.clone(), initial_first.clone()]
		method.affect_analysis(graph, updates, samplers)
		with torch.inference_mode():
			method.layerwise_inference(graph, observer, samplers, (state, None))
		targets = method.collect_nodes[0][-1]
		actual = observer.outputs[targets]
		records.append(
			{
				"implementation": "ICDE-Cache",
				"rows_checked": len(targets),
				"max_error_to_full": float((actual - expected[targets]).abs().max()),
				"intermediate_state_unchanged": bool(torch.equal(state[1], initial_first)),
				"valid": bool(torch.allclose(actual, expected[targets], atol=1e-6, rtol=1e-5)),
			}
		)
		# Match the default frozen configuration: Incremental with model_name='sage'.
		# Its standard model consumes the supplied block, rather than a partial conv.
		method = incremental.Incremental(layers)
		observer = Observer()
		state = [x.clone(), initial_first.clone()]
		aggregates = [value.clone() for value in initial_aggregates]
		method.affect_analysis(graph, updates, samplers)
		with torch.inference_mode():
			method.layerwise_inference(graph, observer, samplers, (state, aggregates))
		targets = method.collect_nodes[0][-1]
		actual = observer.outputs[targets]
		records.append(
			{
				"implementation": "ICDE-Incremental",
				"model_configuration": "default model_name=sage; standard mean SAGEConv",
				"initial_state": "full-model intermediate embeddings and neighbor means",
				"rows_checked": len(targets),
				"max_error_to_full": float((actual - expected[targets]).abs().max()),
				"intermediate_state_unchanged": bool(torch.equal(state[1], initial_first)),
				"aggregate_state_unchanged": all(
					torch.equal(value, original) for value, original in zip(aggregates, initial_aggregates, strict=True)
				),
				"valid": bool(torch.allclose(actual, expected[targets], atol=1e-6, rtol=1e-5)),
			}
		)
	finally:
		dist.destroy_process_group()
	if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
		raise RuntimeError("An upstream implementation changed during verification")
	report = {
		"schema_version": 1,
		"upstream_reference": pin.repository,
		"upstream_revision": pin.revision,
		"implementation_hashes": hashes,
		"unmodified_algorithm": True,
		"adaptations": [
			"Isolated Python package name",
			"World-size-one Gloo process group",
			"Observer captures discarded outputs and dependency IDs",
		],
		"scope": "synthetic kernel verification only; no performance measurement or original end-to-end launch",
		"workload": "mean GraphSAGE; 32 vertices; directed simple graph with explicit self loops; insert edge 0 -> 16",
		"model": {"features": features, "hidden": hidden, "classes": classes, "layers": layers},
		"seed": seed,
		"fixture_sha256": fixture_hash.hexdigest(),
		"dependencies": {"python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__},
		"valid": all(record["valid"] for record in records),
		"records": records,
	}
	(output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
	return report
