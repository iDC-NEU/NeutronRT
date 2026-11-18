"""Conventional DGL full-neighbor and neighbor-sampled inference baselines.

The layerwise bootstrap follows the control flow of DGL v2.4.0's official
GraphSAGE example (Apache-2.0): one native full-neighbor sampler per layer,
with this project's ``GNNModel`` adapter and recorded workload around it.
Selected-output updates use DGL's native multilayer samplers directly. No
incremental engine, custom block compiler, or project-specific sampler is
involved.
"""

import logging
from time import perf_counter
from typing import Iterator, MutableMapping

import dgl
import torch
from torch import Tensor

from lib.graph.capacity import CapacityAdmissionError, CapacityPolicy
from lib.graph.mutation import apply_prepared_mutation, prepare_mutation
from lib.models import GNNModel

OFFICIAL_DGL_RELEASE = "2.4.0"
OFFICIAL_DGL_COMMIT = "c6c874bf7ea085beb04ea1487cfd216a0bacd6c1"
UER_SOURCE_COMMIT = "e6e38887cf94005453d049901010b1d2cc0ce6f7"
UER_SOURCE_PATH = "lib/methods/cache.py"
UER_SOURCE_SHA256 = "57be1de1ef14cc71742c85f2bfd64c74b41fe5cfe9aa01a31065f1d52e9bc829"

_TIMING_KEYS = (
	"native_sampler_block_seconds",
	"transfer_seconds",
	"compute_seconds",
	"output_copy_seconds",
	"intermediate_write_seconds",
)


def _new_timing() -> dict[str, float]:
	return dict.fromkeys(_TIMING_KEYS, 0.0)


def _record(timing: MutableMapping[str, float] | None, key: str, seconds: float) -> None:
	if timing is not None:
		timing[key] = timing.get(key, 0.0) + seconds


def _synchronize(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _validate_official_backend() -> None:
	installed = dgl.__version__.split("+", 1)[0]
	if installed != OFFICIAL_DGL_RELEASE:
		raise RuntimeError(
			f"This standard baseline requires DGL {OFFICIAL_DGL_RELEASE}; installed build is {dgl.__version__!r}"
		)


def _validate_inputs(graph, features: Tensor, model: GNNModel, nodes: Tensor, batch_size: int) -> torch.device:
	_validate_official_backend()
	if batch_size < 1 or model.training:
		raise ValueError("Baseline inference requires a positive batch size and model.eval()")
	if graph.device.type not in ("cpu", "cuda") or features.device != graph.device or nodes.device != graph.device:
		raise ValueError("Baseline topology, features and output node IDs must share one device")
	if graph.device.type == "cuda" and graph.device.index is None:
		raise ValueError("Resident CUDA baselines require an explicit CUDA index")
	if (
		features.ndim != 2
		or features.dtype != torch.float32
		or features.shape[0] != graph.num_nodes()
		or nodes.ndim != 1
		or nodes.dtype != torch.int64
	):
		raise ValueError("Features and output IDs must match the graph")
	return next(model.parameters()).device


def _sampler(model: GNNModel, fanout: int | None):
	if fanout is None:
		return dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
	return dgl.dataloading.NeighborSampler([fanout] * model.num_layers)


def _layer_sampler(fanout: int | None):
	if fanout is None:
		return dgl.dataloading.MultiLayerFullNeighborSampler(1)
	return dgl.dataloading.NeighborSampler([fanout])


def _resident_logical_bound(graph, features: Tensor, model: GNNModel, *, edge_count: int | None = None) -> int:
	"""Conservative native-resident bound shared by Full/NS/UER admission."""
	edges = graph.num_edges() if edge_count is None else edge_count
	index_bytes = 8

	def storage(edge_total: int) -> int:
		return (
			2 * edge_total * index_bytes
			+ 2 * (int(graph.num_nodes()) + 1 + edge_total) * index_bytes
			+ edge_total * index_bytes
		)

	graph_bytes = storage(edges) + storage(graph.num_edges())
	state_bytes = features.numel() * features.element_size()
	state_bytes += graph.num_nodes() * model.num_classes * features.element_size()
	state_bytes += sum(value.numel() * value.element_size() for value in model.parameters())
	state_bytes += graph.num_nodes()  # ready bitmap
	if model.num_layers > 1:
		state_bytes += (
			graph.num_nodes() * sum(model.num_hiddens for _ in range(model.num_layers - 1)) * features.element_size()
		)
	workspace = max(
		1,
		edges * 5 * index_bytes,
		graph.num_nodes() * max(features.shape[1], model.num_hiddens) * features.element_size(),
	)
	return int(graph_bytes + state_bytes + workspace)


def apply_update(graph, features: Tensor, request) -> bool:
	"""Apply one recorded mutation; return whether topology changed."""
	receipt = prepare_mutation(graph, features, request)
	return apply_prepared_mutation(graph, features, receipt)


def infer_batches(
	graph,
	features: Tensor,
	model: GNNModel,
	nodes: Tensor,
	*,
	fanout: int | None,
	batch_size: int,
	sampler=None,
	timing: MutableMapping[str, float] | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
	"""Yield output IDs and embeddings on the model device, without a dense cache.

	``None`` selects full neighborhoods; a positive fanout applies at every layer.
	Topology, features and node IDs share one device. CPU inputs are appropriate
	for the ordinary reference; the resident evaluator supplies a CUDA graph,
	feature table and node metadata so native DGL does not route through CPU rows.
	A supplied native sampler is retained across batches; the streaming adapter
	replaces it after a topology mutation so no stale sampler mapping survives.
	"""
	if fanout is not None and (type(fanout) is not int or fanout < 1):
		raise ValueError("Neighbor-sampling fanout must be positive")
	device = _validate_inputs(graph, features, model, nodes, batch_size)
	fanouts = [fanout if fanout is not None else -1] * model.num_layers
	if sampler is None:
		sampler = _sampler(model, fanout)
	elif sampler.fanouts != fanouts:
		raise ValueError("Retained sampler fanouts differ from the requested baseline")
	loader = dgl.dataloading.DataLoader(
		graph, nodes, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
	)
	iterator = iter(loader)
	while True:
		sampling_started = perf_counter()
		try:
			input_nodes, output_nodes, blocks = next(iterator)
		except StopIteration:
			break
		_record(timing, "native_sampler_block_seconds", perf_counter() - sampling_started)
		transfer_started = perf_counter()
		_synchronize(device)
		with torch.inference_mode():
			h = features[input_nodes].to(device)
			blocks = [block.to(device) for block in blocks]
		_synchronize(device)
		_record(timing, "transfer_seconds", perf_counter() - transfer_started)
		compute_started = perf_counter()
		with torch.inference_mode():
			for layer, block in enumerate(blocks):
				h = model(layer, block, h)
		_synchronize(device)
		_record(timing, "compute_seconds", perf_counter() - compute_started)
		yield output_nodes, h


def infer_layer_batches(
	graph,
	features: Tensor,
	model: GNNModel,
	layer: int,
	nodes: Tensor,
	*,
	fanout: int | None,
	batch_size: int,
	timing: MutableMapping[str, float] | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
	"""Run one native DGL layer over selected output rows.

	This is the shared primitive for the author UER adaptation. It deliberately
	keeps the source feature state CPU-resident and lets DGL construct the
	one-layer block, rather than reproducing block numbering or relabeling.
	"""
	if type(layer) is not int or layer < 0 or layer >= model.num_layers:
		raise ValueError("Layer index is outside the model")
	if fanout is not None and (type(fanout) is not int or fanout < 1):
		raise ValueError("Neighbor-sampling fanout must be positive")
	device = _validate_inputs(graph, features, model, nodes, batch_size)
	loader = dgl.dataloading.DataLoader(
		graph,
		nodes,
		_layer_sampler(fanout),
		batch_size=batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=0,
	)
	iterator = iter(loader)
	while True:
		sampling_started = perf_counter()
		try:
			input_nodes, output_nodes, blocks = next(iterator)
		except StopIteration:
			break
		_record(timing, "native_sampler_block_seconds", perf_counter() - sampling_started)
		transfer_started = perf_counter()
		_synchronize(device)
		with torch.inference_mode():
			h = features[input_nodes].to(device)
			block = blocks[0].to(device)
		_synchronize(device)
		_record(timing, "transfer_seconds", perf_counter() - transfer_started)
		compute_started = perf_counter()
		with torch.inference_mode():
			h = model(layer, block, h)
		_synchronize(device)
		_record(timing, "compute_seconds", perf_counter() - compute_started)
		yield output_nodes, h


def layerwise_infer(
	graph,
	features: Tensor,
	model: GNNModel,
	*,
	batch_size: int,
	progress: bool = False,
) -> tuple[Tensor, dict[str, float | int]]:
	"""Run the official all-node layerwise inference pattern.

	Each layer uses ``MultiLayerFullNeighborSampler(1)`` and a native DGL
	``DataLoader``. Intermediate rows stay on CPU between layers; only one
	minibatch and its one-layer block are placed on the model device.
	"""
	states, timing = layerwise_states(graph, features, model, batch_size=batch_size, progress=progress)
	return states[-1], timing


def layerwise_states(
	graph,
	features: Tensor,
	model: GNNModel,
	*,
	batch_size: int,
	fanout: int | None = None,
	progress: bool = False,
) -> tuple[list[Tensor], dict[str, float | int]]:
	"""Return every layer state using the official DGL layerwise pattern."""
	nodes = torch.arange(graph.num_nodes(), dtype=torch.int64, device=graph.device)
	device = _validate_inputs(graph, features, model, nodes, batch_size)
	h = features
	states = [features]
	timing = _new_timing()
	timing["layers"] = model.num_layers
	timing["batches"] = 0
	temporary_peak = 0
	for layer in range(model.num_layers):
		layer_started = perf_counter()
		sampler = _layer_sampler(fanout)
		loader = dgl.dataloading.DataLoader(
			graph, nodes, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
		)
		output_width = model.num_classes if layer == model.num_layers - 1 else model.num_hiddens
		next_h = torch.empty((graph.num_nodes(), output_width), dtype=h.dtype, device=h.device)
		temporary_peak = max(temporary_peak, h.numel() * h.element_size() + next_h.numel() * next_h.element_size())
		iterator = iter(loader)
		layer_batches = 0
		while True:
			sampling_started = perf_counter()
			try:
				input_nodes, output_nodes, blocks = next(iterator)
			except StopIteration:
				break
			timing["native_sampler_block_seconds"] += perf_counter() - sampling_started
			transfer_started = perf_counter()
			_synchronize(device)
			with torch.inference_mode():
				batch_h = h[input_nodes].to(device)
				block = blocks[0].to(device)
			_synchronize(device)
			timing["transfer_seconds"] += perf_counter() - transfer_started
			compute_started = perf_counter()
			with torch.inference_mode():
				batch_h = model(layer, block, batch_h)
			_synchronize(device)
			timing["compute_seconds"] += perf_counter() - compute_started
			output_started = perf_counter()
			with torch.inference_mode():
				next_h.index_copy_(0, output_nodes, batch_h.to(next_h.device))
			timing["output_copy_seconds"] += perf_counter() - output_started
			layer_batches += 1
			timing["batches"] += 1
			if progress and (layer_batches == 1 or layer_batches % 128 == 0):
				logging.info("DGL layerwise initialization: layer=%d batches=%d", layer, layer_batches)
		h = next_h
		states.append(h)
		timing[f"layer_{layer}_wall_seconds"] = perf_counter() - layer_started
	timing["temporary_intermediate_bytes_peak"] = temporary_peak
	return states, timing


class StreamingRecompute:
	"""Basic affected-output recomputation using native DGL topology operations."""

	def __init__(
		self,
		graph,
		features,
		model,
		*,
		fanout: int | None,
		batch_size: int,
		output_device,
		sampling_seed: int | None = None,
		output_budget_bytes: int | None = None,
		capacity_policy: CapacityPolicy | None = None,
		capacity_estimator=None,
	):
		if model.model_name != "sage":
			raise ValueError("The streaming DGL reference currently supports mean GraphSAGE")
		if fanout is not None and (type(fanout) is not int or fanout < 1):
			raise ValueError("Neighbor-sampling fanout must be positive")
		if type(batch_size) is not int or batch_size < 1:
			raise ValueError("Baseline batch size must be a positive integer")
		if output_budget_bytes is not None and (type(output_budget_bytes) is not int or output_budget_bytes < 1):
			raise ValueError("Output budget must be a positive integer when supplied")
		output_device = torch.device(output_device)
		model_device = next(model.parameters()).device
		if model_device.type not in ("cpu", "cuda") or (model_device.type == "cuda" and model_device.index is None):
			raise ValueError("Model execution must use CPU or an explicit CUDA device")
		if output_device.type == "cuda" and output_budget_bytes is None:
			raise ValueError("CUDA output caches require an explicit output budget")
		if output_device.type not in ("cpu", "cuda") or (output_device.type == "cuda" and output_device.index is None):
			raise ValueError("Output cache must use CPU or an explicit CUDA device")
		if output_device.type == "cuda" and output_device != model_device:
			raise ValueError("CUDA output cache must use the model's CUDA device")
		if graph.device.type == "cuda" and (graph.device != model_device or features.device != graph.device):
			raise ValueError("Native resident execution requires graph, features and model on one device")
		self.graph, self.features, self.model = graph, features, model
		self.capacity_policy = capacity_policy
		self.capacity_estimator = capacity_estimator or (
			lambda edges: _resident_logical_bound(self.graph, self.features, self.model, edge_count=edges)
		)
		self.fanout, self.batch_size = fanout, batch_size
		self.sampling_seed = sampling_seed
		if sampling_seed is not None and (type(sampling_seed) is not int or sampling_seed < 0):
			raise ValueError("Sampling seed must be a nonnegative integer when supplied")
		self.sampler_generation = 0
		self.sampler = _sampler(model, fanout)
		output_bytes = graph.num_nodes() * model.num_classes * features.element_size()
		if output_budget_bytes is not None and output_bytes > output_budget_bytes:
			raise ValueError(
				f"Output cache needs {output_bytes} bytes, exceeding the declared budget {output_budget_bytes}"
			)
		self.outputs = torch.empty(graph.num_nodes(), model.num_classes, device=output_device)
		self.ready = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=graph.device)
		self.last_timing = _new_timing()
		self.initialization_timing: dict[str, float | int] | None = None

	def _invalidate_sampler(self) -> None:
		self.sampler_generation += 1
		self.sampler = _sampler(self.model, self.fanout)

	def synchronize(self):
		device = next(self.model.parameters()).device
		if device.type == "cuda":
			torch.cuda.synchronize(device)

	def compute(self, nodes, *, progress=False):
		completed = 0
		timing = _new_timing()
		for output_nodes, values in infer_batches(
			self.graph,
			self.features,
			self.model,
			nodes,
			fanout=self.fanout,
			batch_size=self.batch_size,
			sampler=self.sampler,
			timing=timing,
		):
			output_started = perf_counter()
			self.outputs[output_nodes.to(self.outputs.device)] = values.to(self.outputs.device)
			timing["output_copy_seconds"] += perf_counter() - output_started
			self.ready[output_nodes.to(self.ready.device)] = True
			completed += output_nodes.numel()
			if progress and (
				completed == output_nodes.numel()
				or completed // self.batch_size % 128 == 0
				or completed == nodes.numel()
			):
				logging.info("DGL initialization: rows=%d/%d", completed, nodes.numel())
		self.last_timing = timing
		return timing

	def _common_mutation(self, request):
		"""Prepare and admit before the one shared authoritative DGL mutation."""
		transfer_started = perf_counter()
		receipt = prepare_mutation(self.graph, self.features, request)
		self.synchronize()
		transfer_seconds = perf_counter() - transfer_started
		admission_started = perf_counter()
		if self.capacity_policy is not None:
			try:
				self.capacity_policy.admit(receipt.next_edge_count, self.capacity_estimator)
			except CapacityAdmissionError as exc:
				raise ValueError(str(exc)) from exc
		admission_seconds = perf_counter() - admission_started
		mutation_started = perf_counter()
		changed_topology = apply_prepared_mutation(self.graph, self.features, receipt)
		self.synchronize()
		mutation_seconds = perf_counter() - mutation_started
		preparation_started = perf_counter()
		if changed_topology:
			self._invalidate_sampler()
		self.synchronize()
		self._mutation_preparation_seconds = perf_counter() - preparation_started
		return receipt, changed_topology, transfer_seconds, admission_seconds, mutation_seconds

	def initialize(self, *, progress: bool = False) -> dict[str, float | int]:
		values, timing = layerwise_infer(
			self.graph, self.features, self.model, batch_size=self.batch_size, progress=progress
		)
		output_started = perf_counter()
		self.outputs.copy_(values.to(self.outputs.device))
		timing["output_copy_seconds"] = timing.get("output_copy_seconds", 0.0) + perf_counter() - output_started
		self.ready.fill_(True)
		self.initialization_timing = timing
		self.last_timing = timing
		return timing

	def query(self, nodes):
		nodes_device = nodes.to(self.ready.device)
		missing = nodes_device[~self.ready[nodes_device]].unique()
		if missing.numel():
			self.compute(missing)
		return self.outputs[nodes_device.to(self.outputs.device)].cpu()

	def update(self, request):
		self.synchronize()
		started = perf_counter()
		receipt, changed_topology, transfer_seconds, admission_seconds, mutation_seconds = self._common_mutation(
			request
		)
		first = receipt.first
		if request.kind == "features":
			targets, hops = receipt.feature_nodes, self.model.num_layers
		elif request.kind in ("insert", "delete"):
			if request.kind == "delete" and not changed_topology:
				finished = perf_counter()
				self.last_timing = _new_timing()
				phases = {
					"input_request_transfer": transfer_seconds,
					"admission": admission_seconds,
					"mutation_index": mutation_seconds,
					"method_preparation": self._mutation_preparation_seconds,
					"affected_delta_planning": 0.0,
					"execution": 0.0,
					"publication_output": 0.0,
				}
				phases["residual"] = finished - started - sum(phases.values())
				return {
					"kind": request.kind,
					"changes": first.numel(),
					"affected_outputs": 0,
					"topology_seconds": mutation_seconds,
					"mutation_index_seconds": mutation_seconds,
					"analysis_seconds": 0.0,
					"compute_and_store_seconds": 0.0,
					**self.last_timing,
					"phase_seconds": phases,
					"publication": {"status": "no_op_absent_delete", "seconds": None},
					"sampler_generation": self.sampler_generation,
					"response_seconds": finished - started,
				}
			targets, hops = receipt.edge_targets, self.model.num_layers - 1
		else:
			raise ValueError("Unsupported recorded update kind")
		mutated = perf_counter()
		for _ in range(hops):
			targets = torch.cat((targets, self.graph.out_edges(targets)[1])).unique()
		analyzed = perf_counter()
		self.ready[targets] = False
		timing = self.compute(targets)
		self.synchronize()
		finished = perf_counter()
		result = {
			"kind": request.kind,
			"changes": first.numel(),
			"affected_outputs": targets.numel(),
			"topology_seconds": mutated - started,
			"mutation_index_seconds": mutation_seconds,
			"analysis_seconds": analyzed - mutated,
			"compute_and_store_seconds": finished - analyzed,
			"native_sampler_block_seconds": timing["native_sampler_block_seconds"],
			"transfer_seconds": timing["transfer_seconds"],
			"compute_seconds": timing["compute_seconds"],
			"output_copy_seconds": timing["output_copy_seconds"],
			"intermediate_write_seconds": timing["intermediate_write_seconds"],
			"phase_seconds": {
				"input_request_transfer": transfer_seconds,
				"admission": admission_seconds,
				"mutation_index": mutation_seconds,
				"method_preparation": self._mutation_preparation_seconds,
				"affected_delta_planning": analyzed - mutated,
				"execution": timing["native_sampler_block_seconds"]
				+ timing["transfer_seconds"]
				+ timing["compute_seconds"],
				"publication_output": timing["output_copy_seconds"],
				"residual": 0.0,
			},
			"publication": {"status": "not_applicable_standalone", "seconds": None},
			"sampler_generation": self.sampler_generation,
			"response_seconds": finished - started,
		}
		result["phase_seconds"]["residual"] = result["response_seconds"] - sum(result["phase_seconds"].values())
		if result["phase_seconds"]["residual"] < -1e-6:
			raise RuntimeError("Native update phase timers overlap")
		return result


class UERRecompute(StreamingRecompute):
	"""Author layerwise UER adaptation of the pinned Cache implementation.

	The retained states are CPU tensors for layers before the final output. Each
	update recomputes only the layerwise affected sets from the current graph and
	uses native DGL one-layer blocks. This is an explicit framework ablation
	adapted from the author's Cache source, not an official DGL or lambdaGrapher
	implementation. Full and NS runners do not retain these cross-update states.
	"""

	def __init__(self, graph, features, model, **kwargs):
		if kwargs.get("fanout") is not None:
			raise ValueError("RTEC-UER is the exact full-neighbor reuse baseline and requires fanout=None")
		super().__init__(graph, features, model, **kwargs)
		self.sampler = _layer_sampler(self.fanout)
		self.states: list[Tensor] | None = None
		self.initialized = False

	def _invalidate_sampler(self) -> None:
		self.sampler_generation += 1
		self.sampler = _layer_sampler(self.fanout)

	def initialize(self, *, progress: bool = False) -> dict[str, float | int]:
		states, timing = layerwise_states(
			self.graph,
			self.features,
			self.model,
			batch_size=self.batch_size,
			fanout=self.fanout,
			progress=progress,
		)
		output_started = perf_counter()
		self.outputs.copy_(states[-1].to(self.outputs.device))
		timing["output_copy_seconds"] = timing.get("output_copy_seconds", 0.0) + perf_counter() - output_started
		self.states = states[:-1]
		self.ready.fill_(True)
		self.initialized = True
		self.initialization_timing = timing
		self.last_timing = timing
		return timing

	def _affected_sets(self, request) -> list[Tensor]:
		if request.kind == "features":
			# Feature rows change their own first-layer term and every outgoing
			# destination; later layers expand that frontier once per layer.
			targets = request.first.to(self.graph.device).unique()
		elif request.kind in ("insert", "delete"):
			# Edge changes first alter only the destination row. The cumulative
			# union below expands downstream dependencies at later layers.
			targets = request.second.to(self.graph.device).unique()
		else:
			raise ValueError("Unsupported recorded update kind")
		affected = []
		for layer in range(self.model.num_layers):
			if request.kind == "features" and layer == 0 and targets.numel():
				targets = torch.cat((targets, self.graph.out_edges(targets)[1])).unique()
			affected.append(targets)
			if layer + 1 < self.model.num_layers and targets.numel():
				targets = torch.cat((targets, self.graph.out_edges(targets)[1])).unique()
		return affected

	def query(self, nodes):
		if not self.initialized:
			self.initialize()
		nodes_device = nodes.to(self.ready.device)
		if not self.ready[nodes_device].all():
			# A failed update never exposes stale output; rebuild all states before
			# a subsequent standalone query rather than publishing a partial cache.
			self.initialize()
		return self.outputs[nodes_device.to(self.outputs.device)].cpu()

	def update(self, request):
		if not self.initialized:
			self.initialize()
		self.synchronize()
		started = perf_counter()
		receipt, changed_topology, transfer_seconds, admission_seconds, mutation_seconds = self._common_mutation(
			request
		)
		mutated = perf_counter()
		if request.kind == "delete" and not changed_topology:
			finished = perf_counter()
			self.last_timing = _new_timing()
			phases = {
				"input_request_transfer": transfer_seconds,
				"admission": admission_seconds,
				"mutation_index": mutation_seconds,
				"method_preparation": self._mutation_preparation_seconds,
				"affected_delta_planning": 0.0,
				"execution": 0.0,
				"publication_output": 0.0,
			}
			phases["residual"] = finished - started - sum(phases.values())
			return {
				"kind": request.kind,
				"changes": request.first.numel(),
				"affected_outputs": 0,
				"affected_by_layer": [0] * self.model.num_layers,
				"topology_seconds": mutation_seconds,
				"mutation_index_seconds": mutation_seconds,
				"analysis_seconds": 0.0,
				"compute_and_store_seconds": 0.0,
				"native_sampler_block_seconds": 0.0,
				"transfer_seconds": 0.0,
				"compute_seconds": 0.0,
				"output_copy_seconds": 0.0,
				"intermediate_write_seconds": 0.0,
				"phase_seconds": phases,
				"publication": {"status": "no_op_absent_delete", "seconds": None},
				"sampler_generation": self.sampler_generation,
				"response_seconds": finished - started,
			}
		affected = self._affected_sets(receipt)
		final_nodes = affected[-1]
		self.ready[final_nodes] = False
		analyzed = perf_counter()
		timing = _new_timing()
		timing["intermediate_write_seconds"] = 0.0
		pending_nodes: list[Tensor] = []
		pending_values: list[Tensor] = []
		try:
			assert self.states is not None
			for layer, targets in enumerate(affected):
				if not targets.numel():
					continue
				for output_nodes, values in infer_layer_batches(
					self.graph,
					self.states[layer],
					self.model,
					layer,
					targets,
					fanout=self.fanout,
					batch_size=self.batch_size,
					timing=timing,
				):
					if layer + 1 < self.model.num_layers:
						write_started = perf_counter()
						self.states[layer + 1].index_copy_(0, output_nodes, values.to(self.states[layer + 1].device))
						timing["intermediate_write_seconds"] += perf_counter() - write_started
					else:
						pending_nodes.append(output_nodes)
						pending_values.append(values.to(self.outputs.device))
			if pending_nodes:
				publication_started = perf_counter()
				self.outputs.index_copy_(0, torch.cat(pending_nodes).to(self.outputs.device), torch.cat(pending_values))
				self.ready[final_nodes] = True
				timing["output_copy_seconds"] += perf_counter() - publication_started
		except Exception:
			self.ready[final_nodes] = False
			raise
		self.last_timing = timing
		self.synchronize()
		finished = perf_counter()
		result = {
			"kind": request.kind,
			"changes": request.first.numel(),
			"affected_outputs": final_nodes.numel(),
			"affected_by_layer": [nodes.numel() for nodes in affected],
			"topology_seconds": mutation_seconds,
			"mutation_index_seconds": mutation_seconds,
			"analysis_seconds": analyzed - mutated,
			"compute_and_store_seconds": finished - analyzed,
			"native_sampler_block_seconds": timing["native_sampler_block_seconds"],
			"transfer_seconds": timing["transfer_seconds"],
			"compute_seconds": timing["compute_seconds"],
			"output_copy_seconds": timing["output_copy_seconds"],
			"intermediate_write_seconds": timing["intermediate_write_seconds"],
			"phase_seconds": {
				"input_request_transfer": transfer_seconds,
				"admission": admission_seconds,
				"mutation_index": mutation_seconds,
				"method_preparation": self._mutation_preparation_seconds,
				"affected_delta_planning": analyzed - mutated,
				"execution": timing["native_sampler_block_seconds"]
				+ timing["transfer_seconds"]
				+ timing["compute_seconds"]
				+ timing["intermediate_write_seconds"],
				"publication_output": timing["output_copy_seconds"],
				"residual": 0.0,
			},
			"publication": {"status": "standalone_final_output_index_copy", "seconds": timing["output_copy_seconds"]},
			"sampler_generation": self.sampler_generation,
			"response_seconds": finished - started,
		}
		result["phase_seconds"]["residual"] = result["response_seconds"] - sum(result["phase_seconds"].values())
		if result["phase_seconds"]["residual"] < -1e-6:
			raise RuntimeError("UER update phase timers overlap")
		return result
