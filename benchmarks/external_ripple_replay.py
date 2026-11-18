"""Standalone Ripple replay adapter for the explicit GC-mean compatibility cohort."""

import argparse
import hashlib
import json
import pickle
import platform
import resource
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import dgl
import dgl.function as fn
import numpy as np
import torch

from benchmarks.ripple import _module
from benchmarks.sources import PinnedSource
from lib.dataset.inference import load_graph
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.tensors import TensorPayload, decode_tensor


def external_gcmean_mapping(model) -> tuple[np.ndarray, ...]:
	"""Map only an exact zero-self, zero-bias mean-SAGE model to Ripple weights."""
	if getattr(model, "model_name", None) != "sage":
		raise ValueError("Ripple GC-mean compatibility requires a mean GraphSAGE model")
	weights = []
	for index, layer in enumerate(model.layers):
		if getattr(layer, "_aggre_type", None) != "mean":
			raise ValueError(f"Layer {index} is not mean aggregation")
		self_weight = layer.fc_self.weight.detach()
		self_bias = layer.fc_self.bias.detach()
		neighbor_bias = layer.fc_neigh.bias
		if torch.count_nonzero(self_weight) or torch.count_nonzero(self_bias):
			raise ValueError("Ordinary GraphSAGE self terms are nonzero; refusing to drop them")
		if neighbor_bias is not None and torch.count_nonzero(neighbor_bias.detach()):
			raise ValueError("GraphSAGE neighbor bias is nonzero; Ripple GC-mean has no bias")
		weights.append(layer.fc_neigh.weight.detach().cpu().numpy().T.copy())
	return tuple(weights)


def _decode(value: wire.Tensor, *, limit: int = 256 * 1024**2) -> torch.Tensor:
	return decode_tensor(TensorPayload(value.dtype, tuple(value.shape), value.data), max_bytes=limit)


def _file_sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for chunk in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _path_sha256(path: Path) -> str | None:
	path = Path(path)
	if path.is_file():
		return _file_sha256(path)
	manifest = path / "manifest.json"
	return _file_sha256(manifest) if path.is_dir() and manifest.is_file() else None


def _write_gcmean_compatibility(recording, graph, features: torch.Tensor, output: Path):
	"""Create an explicit zero-self GC-mean model without editing its source checkpoint."""
	model = recording.spec.build().cpu().eval()
	zeroed = []
	for index, layer in enumerate(model.layers):
		with torch.no_grad():
			layer.fc_self.weight.zero_()
			if layer.fc_self.bias is not None:
				layer.fc_self.bias.zero_()
				zeroed.append(f"layers.{index}.fc_self.bias")
			zeroed.append(f"layers.{index}.fc_self.weight")
			if layer.fc_neigh.bias is not None:
				layer.fc_neigh.bias.zero_()
				zeroed.append(f"layers.{index}.fc_neigh.bias")
	checkpoint = output / "gcmean-compatibility-checkpoint.pt"
	checkpoint_config = {
		"model_name": recording.spec.name,
		"num_features": recording.spec.features,
		"num_hiddens": recording.spec.hidden,
		"num_classes": recording.spec.classes,
		"num_layers": recording.spec.layers,
		"num_heads": recording.spec.heads,
		"seed": recording.spec.seed,
		"relations": recording.spec.relations,
		"pseudo_dim": recording.spec.pseudo_dim,
		"kernels": recording.spec.kernels,
		"aggregator_type": "mean",
	}
	torch.save({"model_config": checkpoint_config, "state_dict": model.state_dict()}, checkpoint)
	spec = replace(recording.spec, checkpoint=str(checkpoint))
	identity = inference_identity(graph, features, spec, model=model)
	compatible = replace(recording, spec=spec, input_identity=identity)
	workload = output / "gcmean-compatibility-workload.pt"
	compatible.save(workload)
	if compatible.digest != recording.digest:
		raise RuntimeError("GC-mean model adaptation changed the recorded update/query sequence")
	return (
		checkpoint,
		workload,
		{
			"mode": "explicit_zero_self_and_bias_compatibility",
			"source_model_identity": recording.input_identity,
			"gcmean_model_identity": identity,
			"original_trace_digest": recording.digest,
			"adapted_trace_digest": compatible.digest,
			"zeroed_parameters": zeroed,
			"derived_checkpoint_sha256": _file_sha256(checkpoint),
			"derived_workload_sha256": _file_sha256(workload),
			"original_checkpoint_sha256": _file_sha256(Path(recording.spec.checkpoint))
			if recording.spec.checkpoint
			else None,
		},
	)


def _simple_graph(graph) -> None:
	sources, targets = graph.edges()
	pairs = list(zip(sources.tolist(), targets.tolist(), strict=True))
	if any(source == target for source, target in pairs):
		raise ValueError("Ripple replay requires a directed simple graph without self-loops")
	if len(pairs) != len(set(pairs)):
		raise ValueError("Ripple replay rejects parallel edges; no deduplication is performed")


def _adjacency(graph) -> tuple[dict[int, set[int]], dict[int, set[int]], set[tuple[int, int]]]:
	outgoing = defaultdict(set)
	incoming = defaultdict(set)
	pairs = set()
	for source, target in zip(*(value.tolist() for value in graph.edges()), strict=True):
		outgoing[source].add(target)
		incoming[target].add(source)
		pairs.add((source, target))
	for node in range(graph.num_nodes()):
		outgoing[node]
		incoming[node]
	return outgoing, incoming, pairs


def _events(request: wire.Update, operations, *, features: int) -> list[Any]:
	first = _decode(request.first)
	second = _decode(request.second)
	if request.HasField("attributes"):
		raise ValueError("Ripple replay does not support edge attributes")
	if first.dtype != torch.int64 or first.ndim != 1:
		raise ValueError("Ripple replay IDs must be int64 vectors")
	if request.kind == wire.Update.FEATURES:
		if second.dtype != torch.float32 or tuple(second.shape) != (first.numel(), features):
			raise ValueError("Feature update rows do not match the graph model width")
		if first.unique().numel() != first.numel():
			raise ValueError("Ripple replay requires unique feature update IDs")
		return [
			operations.GraphStreamEvent(
				index,
				operations.OperationType.NODE_UPDATE,
				node_id=int(node),
				feature=value.detach().cpu().numpy().copy(),
			)
			for index, (node, value) in enumerate(zip(first.tolist(), second, strict=True))
		]
	if request.kind not in (wire.Update.INSERT, wire.Update.DELETE):
		raise ValueError("Ripple replay supports only feature, insert, and delete updates")
	if second.dtype != torch.int64 or second.shape != first.shape:
		raise ValueError("Edge endpoints must be equal-length int64 vectors")
	kind = (
		operations.OperationType.EDGE_ADD
		if request.kind == wire.Update.INSERT
		else operations.OperationType.EDGE_DELETE
	)
	return [
		operations.GraphStreamEvent(index, kind, source_id=int(source), target_id=int(target))
		for index, (source, target) in enumerate(zip(first.tolist(), second.tolist(), strict=True))
	]


def _validate_sequence(batches: list[list[Any]], pairs: set[tuple[int, int]], nodes: int) -> None:
	current = set(pairs)
	for batch in batches:
		for event in batch:
			if event.operation_type.name == "EDGE_ADD":
				pair = (event.source_id, event.target_id)
				if pair in current:
					raise ValueError(f"Ripple replay refuses an existing edge insert: {pair}")
				current.add(pair)
			elif event.operation_type.name == "EDGE_DELETE":
				pair = (event.source_id, event.target_id)
				if pair not in current:
					raise ValueError(f"Ripple replay refuses an absent edge delete: {pair}")
				current.remove(pair)
			elif not 0 <= event.node_id < nodes:
				raise ValueError(f"Ripple replay node ID is out of range: {event.node_id}")


def _apply_oracle(graph, features, batch, operations) -> None:
	for event in batch:
		if event.operation_type == operations.OperationType.EDGE_ADD:
			graph.add_edges(event.source_id, event.target_id)
		elif event.operation_type == operations.OperationType.EDGE_DELETE:
			edge_id = graph.edge_ids(event.source_id, event.target_id)
			graph.remove_edges(edge_id)
		elif event.operation_type == operations.OperationType.NODE_UPDATE:
			features[event.node_id] = torch.from_numpy(event.feature)


def _full(graph, features: torch.Tensor, weights: tuple[np.ndarray, ...]) -> torch.Tensor:
	h = features
	for index, weight in enumerate(weights):
		with graph.local_scope():
			graph.ndata["h"] = h
			graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_neigh"))
			mean = graph.ndata["h_neigh"]
		w = torch.from_numpy(weight).to(dtype=h.dtype)
		h = mean @ w
		if index + 1 < len(weights):
			h = torch.relu(h)
	return h


def _write_state(output: Path, graph, features: torch.Tensor, model, weights: tuple[np.ndarray, ...]) -> int:
	data = output / "data" / "shared"
	state = output / "state"
	data.mkdir(parents=True)
	state.mkdir()
	outgoing, incoming, _ = _adjacency(graph)
	for name, value in (("adj_list", outgoing), ("in_adj_list", incoming)):
		(data / f"{name}.pkl").write_bytes(pickle.dumps(dict(value), protocol=4))
	(state / "weights.pkl").write_bytes(pickle.dumps(list(weights), protocol=4))
	with torch.inference_mode():
		current = features
		for index in range(len(weights) + 1):
			if index:
				current = model.layers[index - 1](graph, current)
				if index < len(weights):
					current = torch.relu(current)
				raw = current.detach().cpu().numpy().copy()
			else:
				raw = current.detach().cpu().numpy().copy()
			np.save(state / f"embeddings_l{index}.npy", raw)
			np.save(state / f"logits_l{index}.npy", raw)
	return sum(path.stat().st_size for root in (data, state) for path in root.rglob("*") if path.is_file())


def _initialize_incremental_messages(processor) -> None:
	"""Seed the author's mutable message cache from the generated initial state."""
	for layer in range(1, processor.config.n_layers + 1):
		messages = processor.messages[layer]
		for target, sources in processor.in_adj_list.items():
			if sources:
				messages[target] = processor.embeddings[layer - 1][list(sources)].sum(axis=0)


def _measured_replay(processor, measured, *, layers: int, validate_batch) -> dict[str, Any]:
	"""Run service intervals and validate between them without charging validation to service time."""
	service_elapsed = 0.0
	loop_wall_start = time.perf_counter()
	updates = []
	correctness = []
	for index, (batch, query) in enumerate(measured):
		publication_start = time.perf_counter()
		processor.process_batch(batch)
		publication_elapsed = time.perf_counter() - publication_start
		query_start = time.perf_counter()
		query_nodes = query.tolist()
		queried = np.asarray(processor.embeddings[layers][query_nodes]).copy()
		query_elapsed = time.perf_counter() - query_start
		service_elapsed += publication_elapsed + query_elapsed
		updates.append(
			{
				"batch": index,
				"input_events": len(batch),
				"query_rows": len(query_nodes),
				"publication_seconds": publication_elapsed,
				"query_seconds": query_elapsed,
				"publication": "author_process_batch_return",
				"queried_sha256": hashlib.sha256(queried.tobytes()).hexdigest(),
			}
		)
		valid, max_abs_error = validate_batch(index, batch, processor)
		correctness.append(
			{
				"batch": index,
				"rows_checked": processor.embeddings[layers].shape[0],
				"max_abs_error": max_abs_error,
				"valid": valid,
			}
		)
		if not valid:
			break
	return {
		"service_elapsed_seconds": service_elapsed,
		"loop_wall_seconds": time.perf_counter() - loop_wall_start,
		"updates": updates,
		"correctness": correctness,
	}


def replay(
	checkout: Path,
	output: Path,
	workload_path: Path,
	graph_path: Path,
	*,
	checkpoint: Path | None = None,
	processor_name: str = "incremental",
	seed: int = 71,
	gcmean_compatibility: bool = False,
	add_inverse_edges: bool = False,
) -> dict[str, Any]:
	if processor_name not in {"incremental", "rc"}:
		raise ValueError("processor must be incremental or rc")
	harness_start = time.perf_counter()
	pin = PinnedSource.load("ripple")
	output.mkdir(parents=True, exist_ok=False)
	source_start = time.perf_counter()
	source = pin.snapshot(checkout, output / "source")
	source_snapshot_seconds = time.perf_counter() - source_start
	data_start = time.perf_counter()
	recording = RecordedWorkload.load(workload_path, checkpoint=checkpoint)
	graph = load_graph(graph_path, add_inverse_edges=add_inverse_edges)
	features = graph.ndata.pop("feat").float().contiguous()
	if inference_identity(graph, features, recording.spec) != recording.input_identity:
		raise ValueError("Recorded workload input identity differs from the supplied graph/model")
	_simple_graph(graph)
	compatibility = None
	if gcmean_compatibility:
		checkpoint, workload_path, compatibility = _write_gcmean_compatibility(recording, graph, features, output)
		recording = RecordedWorkload.load(workload_path, checkpoint=checkpoint)
	data_load_seconds = time.perf_counter() - data_start
	model_start = time.perf_counter()
	model = recording.spec.build()
	weights = external_gcmean_mapping(model)
	model_load_seconds = time.perf_counter() - model_start
	if processor_name == "incremental":
		processor_path = source / "src/workloads/w3_gcn_mean/single/in.py"
	else:
		processor_path = source / "src/workloads/w3_gcn_mean/single/rc.py"
	model_path = source / "src/models/custom_models.py"
	event_path = source / "src/utils/graphstream.py"
	operation_path = source / "src/utils/operations.py"
	hashes = {
		path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
		for path in (model_path, processor_path, event_path, operation_path)
	}
	sys.path[:0] = [str(processor_path.parent), str(source)]
	models = _module("external_ripple_author_models", model_path)
	processor_module = _module("external_ripple_author_processor", processor_path)
	events_module = _module("external_ripple_author_events", event_path)
	operations = type(
		"Operations",
		(),
		{
			"OperationType": sys.modules["src.utils.operations"].OperationType,
			"GraphStreamEvent": events_module.GraphStreamEvent,
		},
	)
	weights_model = models.CustomGraphConvMean(
		recording.spec.features, recording.spec.classes, recording.spec.hidden, recording.spec.layers
	).eval()
	for layer, weight in zip(weights_model.layers, weights, strict=True):
		layer.W.weight.data.copy_(torch.from_numpy(weight.T))
	bootstrap_start = time.perf_counter()
	state_bytes = _write_state(output, graph, features, weights_model, weights)

	config_base: Any = processor_module.Config

	class Config(config_base):
		@property
		def data_root(self):
			return output / "data" / "shared"

		@property
		def model_root(self):
			return output / "state"

		@property
		def embs_root(self):
			return output / "state"

	config = Config("shared", 1, recording.spec.layers, recording.spec.hidden, 0, output / "data", 1)
	processor = processor_module.GraphStreamProcessor(config)
	if processor_name == "incremental":
		_initialize_incremental_messages(processor)
	processor_bootstrap_seconds = time.perf_counter() - bootstrap_start
	oracle_graph = graph.clone()
	oracle_features = features.clone()
	predecode_start = time.perf_counter()
	warmup = [_events(request, operations, features=recording.spec.features) for request in recording.warmup]
	measured = [
		(_events(request, operations, features=recording.spec.features), _decode(query))
		for request, query in recording.steps
	]
	predecode_seconds = time.perf_counter() - predecode_start
	validation_start = time.perf_counter()
	_validate_sequence(warmup + [batch for batch, _ in measured], _adjacency(graph)[2], graph.num_nodes())
	input_validation_seconds = time.perf_counter() - validation_start
	warmup_start = time.perf_counter()
	for batch in warmup:
		processor.process_batch(batch)
		_apply_oracle(oracle_graph, oracle_features, batch, operations)
		if not torch.allclose(
			torch.from_numpy(processor.embeddings[recording.spec.layers]),
			_full(oracle_graph, oracle_features, weights),
			atol=recording.atol,
			rtol=recording.rtol,
		):
			raise ValueError("Ripple author processor failed the warmup full-neighbor oracle")
	warmup_seconds = time.perf_counter() - warmup_start

	def validate_batch(index, batch, current_processor):
		_apply_oracle(oracle_graph, oracle_features, batch, operations)
		expected = _full(oracle_graph, oracle_features, weights)
		actual_all = torch.from_numpy(np.asarray(current_processor.embeddings[recording.spec.layers]))
		error = float((actual_all - expected).abs().max()) if actual_all.numel() else 0.0
		return bool(torch.allclose(actual_all, expected, atol=recording.atol, rtol=recording.rtol)), error

	measured_result = _measured_replay(
		processor,
		measured,
		layers=recording.spec.layers,
		validate_batch=validate_batch,
	)
	updates = measured_result["updates"]
	correctness = measured_result["correctness"]
	service_elapsed = measured_result["service_elapsed_seconds"]
	measured_loop_wall_seconds = measured_result["loop_wall_seconds"]
	full_harness_elapsed = time.perf_counter() - harness_start
	valid = len(correctness) == len(measured) and all(record["valid"] for record in correctness)
	implementation_hashes = hashes
	if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
		raise RuntimeError("Pinned Ripple source changed during replay")
	report = {
		"schema_version": 1,
		"valid": valid,
		"status": "passed" if valid else "validation_failed",
		"upstream_repository": pin.repository,
		"upstream_revision": pin.revision,
		"implementation_hashes": implementation_hashes,
		"unmodified_algorithm": True,
		"operator": {
			"name": "Ripple CustomGraphConvMean",
			"math": "W * mean(incoming features), zero self term and zero bias",
			"mapping": "DGL mean GraphSAGE with every fc_self and bias exactly zero",
		},
		"replay": {
			"performance_replay": True,
			"processor": processor_name,
			"stateful_successive_batches": True,
			"timing_available": True,
			"window": (
				"processor.process_batch plus queried-row materialization; initialization, conversion, "
				"hashing, logging, and oracle validation are outside"
			),
			"timing_scope": "sum_of_uninterrupted_per_batch_service_intervals",
			"continuous_wall_throughput_claim": False,
			"oracle_validation": "runs between service intervals and may affect cache/interference state",
			"publication": "author_process_batch_return",
			"event_conversion": "protobuf decoded and GraphStreamEvent objects predecoded outside the timed window",
		},
		"input_identity": recording.input_identity,
		"trace_digest": recording.digest,
		"model_mapping": {"zero_self_and_bias_proof": True, "widths": recording.spec.widths},
		"gcmean_compatibility": compatibility,
		"graph_options": {"add_inverse_edges": add_inverse_edges},
		"counts": {
			"warmup": len(warmup),
			"measured": len(measured),
			"input_event_count": sum(len(batch) for batch, _ in measured),
			"batch_count": len(measured),
			"query_count": len(measured),
			"query_rows": sum(len(query) for _, query in measured),
		},
		"initialization": {
			"state_bytes": state_bytes,
			"outside_timing": True,
			"source_snapshot_seconds": source_snapshot_seconds,
			"data_load_seconds": data_load_seconds,
			"model_load_seconds": model_load_seconds,
			"processor_bootstrap_seconds": processor_bootstrap_seconds,
			"event_predecode_seconds": predecode_seconds,
			"benchmark_input_validation_seconds": input_validation_seconds,
			"warmup_seconds": warmup_seconds,
			"future_trace_not_used_for_processor_bootstrap": True,
			"pickle_inputs": "adapter-generated adjacency and weights only",
		},
		"timing": {
			"metrics_valid": valid,
			"service_elapsed_seconds": service_elapsed if valid else None,
			"measured_loop_wall_seconds": measured_loop_wall_seconds if valid else None,
			"full_harness_elapsed_seconds": full_harness_elapsed,
			"throughput_events_per_second": (
				sum(len(batch) for batch, _ in measured) / service_elapsed if valid and service_elapsed else None
			),
			"throughput_batches_per_second": len(measured) / service_elapsed if valid and service_elapsed else None,
			"publication_latency_seconds": (
				sum(record["publication_seconds"] for record in updates) if valid else None
			),
			"query_latency_seconds": sum(record["query_seconds"] for record in updates) if valid else None,
			"cpu_only": True,
			"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
			"rss_scope": "harness-inclusive process peak; Linux ru_maxrss",
		},
		"dependencies": {
			"python": platform.python_version(),
			"torch": torch.__version__,
			"dgl": dgl.__version__,
			"numpy": np.__version__,
		},
		"source": {"checkout_pin": pin.revision, "snapshot": str(source)},
		"inputs": {
			"graph": {
				"path": str(graph_path),
				"sha256": _path_sha256(graph_path),
				"identity_sha256": recording.input_identity,
				"identity_scope": "graph, features, model configuration and model weights",
			},
			"workload": {"path": str(workload_path), "sha256": _file_sha256(workload_path)},
			"checkpoint": (
				{"path": str(checkpoint), "sha256": _file_sha256(checkpoint)} if checkpoint is not None else None
			),
		},
		"updates": updates,
		"correctness": correctness,
		"seed": seed,
	}
	(output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
	return report


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", type=Path, required=True)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--workload", type=Path, required=True)
	parser.add_argument("--graph", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--processor", choices=("incremental", "rc"), default="incremental")
	parser.add_argument("--seed", type=int, default=71)
	parser.add_argument("--gcmean-compatibility", action="store_true")
	parser.add_argument("--add-inverse-edges", action="store_true")
	args = parser.parse_args()
	try:
		report = replay(
			args.source.resolve(),
			args.output.resolve(),
			args.workload.resolve(),
			args.graph.resolve(),
			checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
			processor_name=args.processor,
			seed=args.seed,
			gcmean_compatibility=args.gcmean_compatibility,
			add_inverse_edges=args.add_inverse_edges,
		)
		print(json.dumps({"valid": report["valid"], "updates": len(report["updates"])}))
		raise SystemExit(0 if report["valid"] else 1)
	except Exception as error:
		args.output.mkdir(parents=True, exist_ok=True)
		report = {
			"schema_version": 1,
			"valid": False,
			"status": "validation_failed",
			"failure": f"{type(error).__name__}: {error}",
		}
		(args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
		print(json.dumps(report), file=sys.stderr)
		raise SystemExit(1) from None


if __name__ == "__main__":
	main()
