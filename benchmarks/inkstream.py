"""Observe released InkStream SAGE with the authors' unchanged dependency patch."""

import copy
import hashlib
import importlib
import inspect
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.sources import PinnedSource

PYG_VERSION = "2.5.3"
PATCHED_SAGE_SHA256 = "fcb4e0e0e0bcdf5b87c06010d636a97273c8a054601899956ecec4a870329752"


def _observe_outputs(processor, pre, post):
	original_user = processor.user_apply
	original_final = processor.model_config[-1][-1]
	current = {}

	def observe_user(*args, **kwargs):
		result = original_user(*args, **kwargs)
		if args[3] == 1:
			current["node"] = args[4]
			pre[args[4]] = result.detach()
		return result

	def observe_final(value):
		result = original_final(value)
		post[current["node"]] = result.detach().reshape(-1)
		return result

	processor.user_apply = observe_user
	processor.model_config[-1][-1] = observe_final


def verify(checkout: Path, output: Path, *, seed: int = 71) -> dict[str, Any]:
	"""Compare all rows with the authors' own full model after four edge batches.

	This is a CPU gate in an explicitly adapted dependency environment, not the
	Python 3.8 / PyTorch 2.2 Docker environment from the original release.
	"""
	if any(name in sys.modules for name in ("SAGE", "inkstream", "inkstream_sage", "utils", "configs")):
		raise RuntimeError("Run InkStream verification in a fresh process")
	import torch
	import torch_geometric
	from torch_geometric.nn import SAGEConv

	dependency = Path(inspect.getfile(SAGEConv))
	if (
		torch_geometric.__version__ != PYG_VERSION
		or hashlib.sha256(dependency.read_bytes()).hexdigest() != PATCHED_SAGE_SHA256
	):
		raise ValueError("Use PyG 2.5.3 with the authors' unchanged SAGE patch and the pinned dependency hash")
	pin = PinnedSource.load("inkstream")
	output = output.resolve()
	output.mkdir(parents=True, exist_ok=False)
	source = pin.snapshot(checkout, output / "source")
	paths = [
		source / name
		for name in ("SAGE.py", "inkstream.py", "inkstream_sage.py", "EventQueue.py", "docker/sage_conv.patch")
	]
	hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
	os.environ["DYNAMIC_GNN_ROOT"] = str(output)
	sys.path.insert(0, str(source))
	model_module = importlib.import_module("SAGE")
	processor_module = importlib.import_module("inkstream_sage")
	if sys.modules["configs"].device.type != "cpu":
		raise ValueError("Use CUDA_VISIBLE_DEVICES='' for this CPU verification gate")
	torch.set_num_threads(1)
	torch.manual_seed(seed)
	count, features, hidden, classes = 16, 4, 8, 3
	nodes = torch.arange(count)
	edges = torch.stack((nodes.repeat(2), torch.cat(((nodes + 1) % count, (nodes + 3) % count))))
	x = torch.randn(count, features) * 0.1
	model = model_module.SAGE(features, hidden, classes, SimpleNamespace(save_int=True, aggr="mean")).eval()
	with torch.inference_mode():
		initial, _, intermediates = model(x, edges)

	def adjacency(edge_index):
		outgoing, incoming = defaultdict(set), defaultdict(set)
		for source_id, target_id in edge_index.T.tolist():
			outgoing[source_id].add(target_id)
			incoming[target_id].add(source_id)
		return outgoing, incoming

	initial_out, initial_in = adjacency(edges)
	fixture_hash = hashlib.sha256()
	for value in (edges, x, *model.state_dict().values()):
		fixture_hash.update(value.contiguous().numpy().tobytes())
	records = []
	batches = [
		("insert", [(0, 2)], []),
		("converging-inserts", [(0, 2), (5, 2)], []),
		("delete", [], [(0, 1)]),
		("mixed", [(4, 8)], [(4, 5)]),
	]
	for label, inserted, removed in batches:
		final_pairs = [tuple(pair) for pair in edges.T.tolist() if tuple(pair) not in removed] + inserted
		final_edges = torch.tensor(final_pairs).T.contiguous()
		final_out, final_in = adjacency(final_edges)
		with torch.inference_mode():
			expected, _, expected_intermediates = model(x, final_edges)
		state = {
			layer: {
				"before": {node: value["a-"][node].clone() for node in range(count)},
				"after": {node: value["a"][node].clone() for node in range(count)},
			}
			for layer, value in intermediates.items()
		}
		processor = processor_module.inkstream_sage(
			model, str(output), aggregator="mean", verify=False, out_channels=classes
		)
		post, pre = initial.clone(), initial.clone()

		# Observe discarded outputs while retaining the original transformation.
		_observe_outputs(processor, pre, post)
		processor.incremental_inference(
			copy.deepcopy(initial_out),
			copy.deepcopy(initial_in),
			copy.deepcopy(final_out),
			copy.deepcopy(final_in),
			state,
			inserted,
			removed,
		)
		hidden_error = max(
			float((state["layer2"]["before"][node] - expected_intermediates["layer2"]["a-"][node]).abs().max())
			for node in range(count)
		)
		records.append(
			{
				"batch": label,
				"inserted_edges": inserted,
				"removed_edges": removed,
				"rows_checked": count,
				"max_output_error": float((post - expected).abs().max()),
				"max_pre_final_activation_error": float((pre - expected).abs().max()),
				"max_hidden_error": hidden_error,
				"valid": bool(torch.allclose(post, expected, atol=1e-6, rtol=1e-5)),
			}
		)
	if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
		raise RuntimeError("An upstream implementation changed during verification")
	report = {
		"schema_version": 1,
		"upstream_repository": pin.repository,
		"upstream_revision": pin.revision,
		"unmodified_algorithm": True,
		"implementation_hashes": hashes,
		"scope": "synthetic numerical verification; adapted dependencies; not authors Docker reproduction or timing",
		"adaptations": [
			"Generated in-memory adjacency/intermediate fixtures",
			"Observers capture discarded outputs",
			"Authors SAGE patch applied unchanged to isolated PyG 2.5.3",
		],
		"workload": "authors mean GraphSAGE; directed simple graph; each batch starts from the same initial state",
		"model": {"features": features, "hidden": hidden, "classes": classes, "layers": 2},
		"seed": seed,
		"fixture_sha256": fixture_hash.hexdigest(),
		"dependencies": {
			"python": platform.python_version(),
			"torch": torch.__version__,
			"torch_geometric": torch_geometric.__version__,
		},
		"patched_sage_sha256": PATCHED_SAGE_SHA256,
		"valid": all(record["valid"] for record in records),
		"records": records,
	}
	(output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
	return report
