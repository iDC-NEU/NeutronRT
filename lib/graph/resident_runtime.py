"""Single-device resident execution for exact mean-GraphSAGE updates.

The runtime owns the mutable DGL graph, feature rows, model weights, full
embedding rows, and projected aggregate rows on one explicit device.  CPU
callers provide only control metadata and raw update/query payloads; no
managed runtime, process transport, or CPU feature lookup is involved.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import dgl  # noqa: F401 - kept as the explicit forbidden-warm-path test seam
import dgl.function as fn
import torch
from torch import Tensor

from lib.graph.capacity import CapacityAdmissionError, CapacityPolicy
from lib.graph.deltas import NeighborUpdate, RoutedRows
from lib.graph.mutation import apply_prepared_mutation, prepare_mutation, prepare_structural_delta
from lib.models import GNNModel
from lib.models.aggregates import delta_sage_state, full_sage_state  # noqa: F401


class ResidentCapacityError(MemoryError):
	"""The declared resident budget cannot admit the requested state."""


@dataclass(frozen=True)
class ResidentAdmission:
	device: str
	capacity_bytes: int
	estimated_bytes: int
	resident_bytes: int


def _graph_storage_bytes(nodes: int, edges: int) -> int:
	"""Bound COO, CSR, CSC, endpoint and edge-id index storage."""
	index_bytes = 8
	coo = 2 * edges * index_bytes
	csr = (nodes + 1 + edges) * index_bytes
	csc = (nodes + 1 + edges) * index_bytes
	eids = edges * index_bytes
	return coo + csr + csc + eids


def estimate_resident_bytes(
	graph,
	features: Tensor,
	model: GNNModel,
	*,
	aggregate_dtype: torch.dtype | None = None,
	edge_count: int | None = None,
	transition_edge_count: int | None = None,
) -> int:
	"""Conservatively account resident indexes, state, and one update transition."""
	if features.ndim != 2 or features.dtype not in (torch.float32, torch.float64):
		raise ValueError("Resident execution requires a floating-point feature matrix")
	if graph.num_nodes() != features.shape[0]:
		raise ValueError("Resident graph and features have different node counts")
	model_dtype = next(model.parameters()).dtype
	if model_dtype not in (torch.float32, torch.float64):
		raise ValueError("Resident execution requires float32 or float64 model weights")
	aggregate_dtype = aggregate_dtype or model_dtype
	if aggregate_dtype not in (torch.float32, torch.float64):
		raise ValueError("Resident aggregate state requires float32 or float64")
	edges = graph.num_edges() if edge_count is None else edge_count
	if type(edges) is not int or edges < 0:
		raise ValueError("Resident edge count must be a nonnegative integer")
	if transition_edge_count is not None and (type(transition_edge_count) is not int or transition_edge_count < 0):
		raise ValueError("Resident transition edge count must be a nonnegative integer")
	graph_bytes = _graph_storage_bytes(graph.num_nodes(), edges)
	transition_bytes = (
		_graph_storage_bytes(graph.num_nodes(), transition_edge_count) if transition_edge_count is not None else 0
	)
	state_bytes = graph.num_nodes() * features.shape[1] * features.element_size()
	degree_bytes = graph.num_nodes() * torch.int64.itemsize
	state_bytes += (
		graph.num_nodes()
		* sum(model.num_classes if i == model.num_layers - 1 else model.num_hiddens for i in range(model.num_layers))
		* model_dtype.itemsize
	)
	aggregate_bytes = (
		graph.num_nodes()
		* sum(model.num_classes if i == model.num_layers - 1 else model.num_hiddens for i in range(model.num_layers))
		* aggregate_dtype.itemsize
	)
	model_bytes = sum(value.numel() * value.element_size() for value in model.parameters())
	max_edges = max(edges, transition_edge_count or 0)
	# Edge routing, signed weights, compact update rows, one old/new index set,
	# and one layer's scatter scratch. This is a logical admission bound, not a
	# claim that Python/DGL allocator overhead is exactly represented.
	workspace_bytes = max(
		1,
		max_edges * 5 * 8,
		graph.num_nodes()
		* max(features.shape[1], model.num_hiddens)
		* max(model_dtype.itemsize, aggregate_dtype.itemsize),
	)
	return int(
		graph_bytes + transition_bytes + degree_bytes + state_bytes + aggregate_bytes + model_bytes + workspace_bytes
	)


class ResidentGraphSAGE:
	"""Exact resident NrtInc path for fixed mean-GraphSAGE models.

	The bootstrap computes complete state once. Each mutation then routes compact
	old/new source rows over the resident outgoing index and applies the shared
	``S'=S+A'(H'-H)+(A'-A)H`` algebra to affected rows in-place. Warm updates do
	not build full-neighbor blocks or enumerate incoming neighborhoods.
	"""

	def __init__(
		self,
		graph,
		features: Tensor,
		model: GNNModel,
		*,
		device: str | torch.device,
		capacity_bytes: int,
		aggregate_dtype: torch.dtype | None = None,
	) -> None:
		self.device = torch.device(device)
		if self.device.type not in ("cpu", "cuda") or self.device.type == "cuda" and self.device.index is None:
			raise ValueError("Resident execution requires an explicit CPU or CUDA device")
		if type(capacity_bytes) is not int or capacity_bytes < 1:
			raise ValueError("Resident execution requires a positive capacity_bytes")
		if graph.device.type != "cpu" or features.device.type != "cpu":
			raise ValueError("Resident bootstrap expects the authoritative graph and features on CPU")
		if (
			model.training
			or model.model_name != "sage"
			or any(getattr(layer, "_aggre_type", None) != "mean" for layer in model.layers)
		):
			raise ValueError("Resident execution currently supports eval-mode mean GraphSAGE only")
		self.aggregate_dtype = aggregate_dtype or next(model.parameters()).dtype
		if self.aggregate_dtype not in (torch.float32, torch.float64):
			raise ValueError("Resident aggregate state requires float32 or float64")
		estimated = estimate_resident_bytes(
			graph,
			features,
			model,
			aggregate_dtype=self.aggregate_dtype,
			transition_edge_count=graph.num_edges(),
		)
		try:
			self.capacity_policy = CapacityPolicy.create(
				capacity_bytes,
				self.device,
				estimated,
				graph.num_edges(),
				lambda edges: estimate_resident_bytes(
					graph,
					features,
					model,
					aggregate_dtype=self.aggregate_dtype,
					edge_count=edges,
					transition_edge_count=graph.num_edges(),
				),
			)
		except CapacityAdmissionError as exc:
			raise ResidentCapacityError(str(exc)) from exc
		self.capacity_bytes = capacity_bytes
		self.admission = ResidentAdmission(str(self.device), capacity_bytes, estimated, 0)
		self.graph = graph.to(self.device)
		self.features = features.to(self.device)
		self.model = model.to(self.device).eval()
		self.degrees = self.graph.in_degrees().to(self.device)
		self.states: list[Tensor] = [self.features]
		self.aggregates: list[Tensor] = []
		self.version = 0
		self.initialized = False
		self.last_update: dict[str, Any] | None = None
		self.last_query: dict[str, Any] | None = None
		self.counters = {
			"delta_input_rows": 0,
			"routed_messages": 0,
			"full_refreshes": 0,
			"full_neighbor_tasks": 0,
			"warm_updates": 0,
		}
		self._refresh_admission_bytes()

	def _sync(self) -> None:
		if self.device.type == "cuda":
			torch.cuda.synchronize(self.device)

	def _refresh_admission_bytes(self) -> None:
		graph_bytes = _graph_storage_bytes(self.graph.num_nodes(), self.graph.num_edges())
		degree_bytes = self.degrees.numel() * self.degrees.element_size()
		state_bytes = sum(value.numel() * value.element_size() for value in self.states)
		aggregate_bytes = sum(value.numel() * value.element_size() for value in self.aggregates)
		model_bytes = sum(value.numel() * value.element_size() for value in self.model.parameters())
		estimated = estimate_resident_bytes(
			self.graph,
			self.features,
			self.model,
			aggregate_dtype=self.aggregate_dtype,
			transition_edge_count=self.graph.num_edges(),
		)
		self.admission = ResidentAdmission(
			str(self.device),
			self.capacity_bytes,
			estimated,
			graph_bytes + degree_bytes + state_bytes + aggregate_bytes + model_bytes,
		)

	@torch.inference_mode()
	def _bootstrap_layer(self, layer: int, inputs: Tensor) -> tuple[Tensor, Tensor]:
		operator = self.model.layers[layer]
		neighbor = operator.fc_neigh
		if neighbor.bias is not None:
			raise ValueError("Resident projected aggregate state requires bias-free neighbor weights")
		with self.graph.local_scope():
			self.graph.srcdata["_nrt_projected"] = neighbor(inputs)
			if self.graph.num_edges():
				self.graph.update_all(fn.copy_u("_nrt_projected", "_nrt_message"), fn.sum("_nrt_message", "_nrt_sum"))
				sums = self.graph.dstdata["_nrt_sum"]
			else:
				sums = inputs.new_zeros((self.graph.num_nodes(), neighbor.out_features))
			degrees = self.graph.in_degrees().clamp_min(1).to(self.device)
			mean = (sums / degrees.unsqueeze(1)).to(neighbor.weight.dtype)
			output = self.model.from_aggrs(layer, inputs, mean, projected=True)
		return output, sums.to(self.aggregate_dtype)

	@torch.inference_mode()
	def bootstrap(self) -> dict[str, float | int]:
		started = perf_counter()
		self._sync()
		self.states = [self.features]
		self.aggregates = []
		for layer in range(self.model.num_layers):
			output, sums = self._bootstrap_layer(layer, self.states[-1])
			self.aggregates.append(sums)
			self.states.append(output)
		self._sync()
		self.initialized = True
		self.counters["full_refreshes"] += 1
		self.counters["full_neighbor_tasks"] += self.model.num_layers
		self._refresh_admission_bytes()
		return {"seconds": perf_counter() - started, "resident_bytes": self.admission.resident_bytes}

	@torch.inference_mode()
	def _admit_transition(self, next_edges: int) -> None:
		try:
			self.capacity_policy.admit(
				next_edges,
				lambda edges: estimate_resident_bytes(
					self.graph,
					self.features,
					self.model,
					aggregate_dtype=self.aggregate_dtype,
					edge_count=edges,
					transition_edge_count=self.graph.num_edges(),
				),
			)
		except CapacityAdmissionError as exc:
			raise ResidentCapacityError(str(exc)) from exc

	@torch.inference_mode()
	def _capture_old_state(self, receipt):
		"""Capture only rows needed by the delta algebra, before shared mutation."""
		if receipt.kind == "features":
			if receipt.feature_nodes.numel() != receipt.first.numel():
				raise ValueError("Feature mutation node IDs must be unique")
			return (
				receipt.feature_nodes,
				self.features.index_select(0, receipt.feature_nodes),
				self.degrees.new_empty(0),
				receipt.edge_sources,
				receipt.edge_targets,
				receipt.edge_weights,
				[state.new_empty((0, state.shape[1])) for state in self.states],
			)
		if not receipt.edge_weights.numel():
			empty = self.degrees.new_empty(0)
			return (
				empty,
				empty,
				empty,
				empty,
				empty,
				empty,
				[state.new_empty((0, state.shape[1])) for state in self.states],
			)
		old_degrees = self.degrees.index_select(0, receipt.edge_targets)
		old_edge_values = [state.index_select(0, receipt.edge_sources) for state in self.states]
		return (
			self.degrees.new_empty(0),
			self.degrees.new_empty(0),
			old_degrees,
			receipt.edge_sources,
			receipt.edge_targets,
			receipt.edge_weights,
			old_edge_values,
		)

	@torch.inference_mode()
	def _merge_old_rows(
		self, nodes: Tensor, values: Tensor, new_nodes: Tensor, new_values: Tensor
	) -> tuple[Tensor, Tensor]:
		if not new_nodes.numel():
			return nodes, values
		if not nodes.numel():
			return new_nodes, new_values
		merged = torch.cat((nodes, new_nodes)).unique(sorted=True)
		result = values.new_empty((merged.numel(), values.shape[1]))
		result.index_copy_(0, torch.searchsorted(merged, nodes), values)
		result.index_copy_(0, torch.searchsorted(merged, new_nodes), new_values)
		return merged, result

	@torch.inference_mode()
	def _sparse_layer_update(
		self,
		layer: int,
		changed_sources: Tensor,
		old_source_values: Tensor,
		edge_sources: Tensor,
		edge_targets: Tensor,
		edge_weights: Tensor,
		old_edge_values: Tensor,
		old_structural_degrees: Tensor,
	) -> tuple[Tensor, Tensor, Tensor, float, float, int, int]:
		"""Apply one S'=S+A'(H'-H)+(A'-A)H update without incoming traversal."""
		planning_started = perf_counter()
		if changed_sources.numel():
			out_sources, out_targets = self.graph.out_edges(changed_sources)
		else:
			out_sources = torch.empty(0, dtype=torch.int64, device=self.device)
			out_targets = torch.empty(0, dtype=torch.int64, device=self.device)
		targets = torch.cat((changed_sources, out_targets, edge_targets)).unique(sorted=True)
		if not targets.numel():
			empty_output = self.states[layer + 1].new_empty((0, self.states[layer + 1].shape[1]))
			empty_sum = self.aggregates[layer].new_empty((0, self.aggregates[layer].shape[1]))
			self._sync()
			return targets, empty_output, empty_sum, perf_counter() - planning_started, 0.0, 0, 0

		input_values = old_source_values
		if changed_sources.numel():
			input_values = self.states[layer].index_select(0, changed_sources) - old_source_values
			input_route = RoutedRows(
				input_values,
				torch.searchsorted(changed_sources, out_sources),
				torch.searchsorted(targets, out_targets),
			)
		else:
			input_route = RoutedRows(
				self.states[layer].new_empty((0, self.states[layer].shape[1])),
				torch.empty(0, dtype=torch.int64, device=self.device),
				torch.empty(0, dtype=torch.int64, device=self.device),
			)
		if edge_sources.numel():
			edge_source_table, edge_source_inverse = torch.unique(edge_sources, sorted=True, return_inverse=True)
			first_source = torch.cat(
				(
					torch.ones(1, dtype=torch.bool, device=self.device),
					edge_sources[1:] != edge_sources[:-1],
				)
			)
			compact_edge_values = old_edge_values[first_source]
			edge_route = RoutedRows(
				compact_edge_values,
				edge_source_inverse,
				torch.searchsorted(targets, edge_targets),
				edge_weights,
			)
		else:
			edge_route = RoutedRows(
				self.states[layer].new_empty((0, self.states[layer].shape[1])),
				torch.empty(0, dtype=torch.int64, device=self.device),
				torch.empty(0, dtype=torch.int64, device=self.device),
				torch.empty(0, dtype=torch.int64, device=self.device),
			)
		old_degrees = self.degrees.index_select(0, targets)
		if edge_targets.numel():
			old_degree_positions = torch.searchsorted(targets, edge_targets)
			old_degrees.index_copy_(0, old_degree_positions, old_structural_degrees)
		self._sync()
		planning_seconds = perf_counter() - planning_started
		neighbor_state = NeighborUpdate(self.aggregates[layer].index_select(0, targets), old_degrees)
		execution_started = perf_counter()
		output, updated = delta_sage_state(
			self.model,
			layer + 1,
			self.states[layer].index_select(0, targets),
			neighbor_state,
			input_route,
			edge_route,
		)
		self._sync()
		return (
			targets,
			output,
			updated.sums,
			planning_seconds,
			perf_counter() - execution_started,
			int(input_route.sources.numel()),
			int(input_route.sources.numel() + edge_route.sources.numel()),
		)

	@torch.inference_mode()
	def update(self, request) -> dict[str, Any]:
		if not self.initialized:
			self.bootstrap()
		self._sync()
		started = perf_counter()
		transfer_started = perf_counter()
		receipt = prepare_mutation(self.graph, self.features, request)
		self._sync()
		transfer_seconds = perf_counter() - transfer_started
		admission_started = perf_counter()
		if receipt.next_edge_count < 0:
			raise ResidentCapacityError("Resident topology transition would create a negative edge count")
		self._admit_transition(receipt.next_edge_count)
		admission_seconds = perf_counter() - admission_started
		prep_started = perf_counter()
		receipt = prepare_structural_delta(receipt)
		(
			feature_nodes,
			old_features,
			old_structural_degrees,
			edge_sources,
			edge_targets,
			edge_weights,
			old_edge_values,
		) = self._capture_old_state(receipt)
		self._sync()
		method_preparation_seconds = perf_counter() - prep_started
		mutation_started = perf_counter()
		apply_prepared_mutation(self.graph, self.features, receipt)
		self._sync()
		mutation_seconds = perf_counter() - mutation_started
		prep_started = perf_counter()
		if receipt.edge_targets.numel():
			self.degrees.index_add_(0, receipt.edge_targets, receipt.edge_weights)
		self._sync()
		method_preparation_seconds += perf_counter() - prep_started
		saved_nodes = [torch.empty(0, dtype=torch.int64, device=self.device) for _ in self.states]
		saved_values = [
			self.states[layer].new_empty((0, self.states[layer].shape[1])) for layer in range(len(self.states))
		]
		if edge_sources.numel():
			for layer in range(len(self.states)):
				saved_nodes[layer] = edge_sources
				saved_values[layer] = old_edge_values[layer]
		if feature_nodes.numel():
			saved_nodes[0], saved_values[0] = self._merge_old_rows(
				saved_nodes[0], saved_values[0], feature_nodes, old_features
			)
		changed_sources = feature_nodes
		affected_by_layer = []
		input_rows = 0
		routed_messages = 0
		planning_seconds = 0.0
		execution_seconds = 0.0
		publication_seconds = 0.0
		for layer in range(self.model.num_layers):
			if changed_sources.numel():
				old_source_values = saved_values[layer].index_select(
					0, torch.searchsorted(saved_nodes[layer], changed_sources)
				)
			else:
				old_source_values = self.states[layer].new_empty((0, self.states[layer].shape[1]))
			targets, output, sums, layer_planning, layer_execution, layer_input_rows, layer_messages = (
				self._sparse_layer_update(
					layer,
					changed_sources,
					old_source_values,
					edge_sources,
					edge_targets,
					edge_weights,
					old_edge_values[layer]
					if edge_sources.numel()
					else self.states[layer].new_empty((0, self.states[layer].shape[1])),
					old_structural_degrees,
				)
			)
			planning_seconds += layer_planning
			execution_seconds += layer_execution
			input_rows += layer_input_rows
			routed_messages += layer_messages
			affected_by_layer.append(int(targets.numel()))
			if not targets.numel():
				continue
			old_outputs = self.states[layer + 1].index_select(0, targets)
			saved_nodes[layer + 1], saved_values[layer + 1] = self._merge_old_rows(
				saved_nodes[layer + 1], saved_values[layer + 1], targets, old_outputs
			)
			publish_started = perf_counter()
			self.aggregates[layer].index_copy_(0, targets, sums)
			self.states[layer + 1].index_copy_(0, targets, output)
			self._sync()
			publication_seconds += perf_counter() - publish_started
			changed_sources = targets
		self.counters["delta_input_rows"] += input_rows
		self.counters["routed_messages"] += routed_messages
		self.counters["warm_updates"] += 1
		self.version += 1
		self._refresh_admission_bytes()
		response = perf_counter() - started
		phases = {
			"input_request_transfer": transfer_seconds,
			"admission": admission_seconds,
			"method_preparation": method_preparation_seconds,
			"mutation_index": mutation_seconds,
			"affected_delta_planning": planning_seconds,
			"execution": execution_seconds,
			"publication_output": publication_seconds,
		}
		phases["residual"] = response - sum(phases.values())
		if phases["residual"] < -1e-6:
			raise RuntimeError("Resident update phase timers overlap")
		self.last_update = {
			"kind": request.kind,
			"changes": int(request.first.numel()),
			"affected_outputs": affected_by_layer[-1] if affected_by_layer else 0,
			"affected_by_layer": affected_by_layer,
			"delta_input_rows": input_rows,
			"routed_messages": routed_messages,
			"full_refreshes": 0,
			"full_neighbor_tasks": 0,
			"counters": dict(self.counters),
			"phase_seconds": phases,
			"response_seconds": response,
			"version": self.version,
			"publication": "resident_state_index_copy",
		}
		return self.last_update

	@torch.inference_mode()
	def query(self, nodes: Tensor) -> Tensor:
		if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
			raise ValueError("Resident queries require CPU int64 node IDs")
		if not self.initialized:
			self.bootstrap()
		started = perf_counter()
		self._sync()
		resident_nodes = nodes.to(self.device, non_blocking=self.device.type == "cuda")
		values = self.states[-1].index_select(0, resident_nodes)
		result = values.cpu()
		self._sync()
		self.last_query = {"rows": int(nodes.numel()), "response_seconds": perf_counter() - started}
		return result
