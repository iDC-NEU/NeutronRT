"""Shared authoritative DGL mutation preparation and application."""

from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreparedMutation:
	"""Validated device-local mutation; sparse signed deltas are optional."""

	kind: str
	first: Tensor
	second: Tensor
	feature_nodes: Tensor
	edge_sources: Tensor
	edge_targets: Tensor
	edge_weights: Tensor
	edge_ids: Tensor
	next_edge_count: int
	topology_changed: bool


def _empty(device: torch.device) -> Tensor:
	return torch.empty(0, dtype=torch.int64, device=device)


@torch.inference_mode()
def prepare_mutation(graph, features: Tensor, request) -> PreparedMutation:
	"""Copy, validate, and resolve one request without mutating authoritative state."""
	device = graph.device
	first = request.first.to(device, non_blocking=device.type == "cuda")
	second = request.second.to(device, non_blocking=device.type == "cuda")
	if request.kind == "features":
		if (
			first.dtype != torch.int64
			or second.dtype != features.dtype
			or first.ndim != 1
			or second.shape != (first.numel(), features.shape[1])
		):
			raise ValueError("Feature mutation rows do not match the feature matrix")
		return PreparedMutation(
			request.kind,
			first,
			second,
			first.unique(sorted=True),
			_empty(device),
			_empty(device),
			_empty(device),
			_empty(device),
			graph.num_edges(),
			False,
		)
	if first.dtype != torch.int64 or second.dtype != torch.int64 or first.ndim != 1 or first.shape != second.shape:
		raise ValueError("Edge mutation endpoints must be aligned int64 vectors")
	if request.kind == "insert":
		return PreparedMutation(
			request.kind,
			first,
			second,
			_empty(device),
			first,
			second,
			_empty(device),
			_empty(device),
			graph.num_edges() + int(first.numel()),
			bool(first.numel()),
		)
	if request.kind != "delete":
		raise ValueError("Unsupported recorded update kind")
	if not first.numel():
		return PreparedMutation(
			request.kind,
			first,
			second,
			_empty(device),
			_empty(device),
			_empty(device),
			_empty(device),
			_empty(device),
			graph.num_edges(),
			False,
		)
	present = graph.has_edges_between(first, second)
	if not present.any():
		return PreparedMutation(
			request.kind,
			first,
			second,
			_empty(device),
			_empty(device),
			_empty(device),
			_empty(device),
			_empty(device),
			graph.num_edges(),
			False,
		)
	eids = graph.edge_ids(first[present], second[present], return_uv=True)[2].unique()
	matched_sources, matched_targets = graph.find_edges(eids)
	return PreparedMutation(
		request.kind,
		first,
		second,
		_empty(device),
		matched_sources,
		matched_targets,
		_empty(device),
		eids,
		graph.num_edges() - int(eids.numel()),
		bool(eids.numel()),
	)


@torch.inference_mode()
def prepare_structural_delta(receipt: PreparedMutation) -> PreparedMutation:
	"""Coalesce signed messages for incremental methods, outside common mutation."""
	if receipt.kind == "features" or not receipt.edge_sources.numel():
		return receipt
	pairs = torch.stack((receipt.edge_sources, receipt.edge_targets), dim=1)
	unique, counts = torch.unique(pairs, dim=0, return_counts=True)
	return replace(
		receipt,
		edge_sources=unique[:, 0].contiguous(),
		edge_targets=unique[:, 1].contiguous(),
		edge_weights=counts.to(torch.int64) * (1 if receipt.kind == "insert" else -1),
	)


@torch.inference_mode()
def apply_prepared_mutation(
	graph,
	features: Tensor,
	receipt: PreparedMutation,
	*,
	degree_index: Tensor | None = None,
	on_topology_changed: Callable[[], None] | None = None,
) -> bool:
	"""Apply a prepared receipt using standard DGL mutation operations."""
	if receipt.first.device != graph.device or features.device != graph.device:
		raise ValueError("Prepared mutation and authoritative state must share a device")
	if receipt.kind == "features":
		features.index_copy_(0, receipt.first, receipt.second)
	elif receipt.kind == "insert":
		if receipt.first.numel():
			graph.add_edges(receipt.first, receipt.second)
	elif receipt.kind == "delete":
		if receipt.edge_ids.numel():
			graph.remove_edges(receipt.edge_ids)
	else:
		raise ValueError("Unsupported prepared mutation kind")
	if degree_index is not None and receipt.edge_targets.numel():
		degree_index.index_add_(0, receipt.edge_targets, receipt.edge_weights)
		if (degree_index < 0).any():
			raise RuntimeError("Topology transition created a negative degree")
	if receipt.topology_changed and on_topology_changed is not None:
		on_topology_changed()
	return receipt.topology_changed
