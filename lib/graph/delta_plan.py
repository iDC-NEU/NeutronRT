"""Storage-side routing metadata for sparse mean-GraphSAGE aggregate updates.

The caller holds a topology fence and supplies signed edge multiplicity changes.
This module plans one version transition; it does not certify aggregate freshness
or implement the worker state/publication protocol.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch import Tensor

from lib.graph.deltas import RoutedRows
from lib.graph.recompute import PlanTooLarge


class OutgoingTopology(Protocol):
	def num_nodes(self) -> int: ...
	def out_degrees(self, nodes: Tensor) -> Tensor: ...
	def out_edges(self, nodes: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class MessagePlan:
	value_nodes: Tensor
	sources: Tensor
	targets: Tensor
	weights: Optional[Tensor]

	def bind(self, values: Tensor) -> RoutedRows:
		"""Values must follow value_nodes; no per-edge vector expansion is retained."""
		if values.ndim != 2 or values.shape[0] != self.value_nodes.numel():
			raise ValueError("Source values must follow the planned source-node table")
		return RoutedRows(
			values,
			self.sources.to(values.device),
			self.targets.to(values.device),
			None if self.weights is None else self.weights.to(values.device),
		)


@dataclass(frozen=True)
class SparseLayerPlan:
	targets: Tensor
	inputs: MessagePlan
	edges: MessagePlan


def select_delta_routes(plan: SparseLayerPlan, positions: Tensor) -> SparseLayerPlan:
	"""Select arbitrary unique target positions and compact both source tables."""
	if (
		positions.device.type != "cpu"
		or positions.dtype != torch.int64
		or positions.ndim != 1
		or positions.unique().numel() != positions.numel()
		or positions.numel()
		and (positions.min() < 0 or positions.max() >= plan.targets.numel())
	):
		raise ValueError("Sparse target positions must be unique CPU int64 indices inside the plan")
	remap = torch.full_like(plan.targets, -1)
	remap[positions] = torch.arange(positions.numel())

	def select(message: MessagePlan) -> MessagePlan:
		targets = remap[message.targets]
		mask = targets >= 0
		used, sources = message.sources[mask].unique(sorted=True, return_inverse=True)
		return MessagePlan(
			message.value_nodes[used],
			sources,
			targets[mask],
			None if message.weights is None else message.weights[mask],
		)

	return SparseLayerPlan(plan.targets[positions], select(plan.inputs), select(plan.edges))


def slice_delta_routes(plan: SparseLayerPlan, start: int, stop: int) -> SparseLayerPlan:
	"""Keep a target interval and compact its source tables without copying values."""
	if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= plan.targets.numel():
		raise ValueError("Sparse target interval is outside its plan")

	def select(message: MessagePlan) -> MessagePlan:
		mask = (message.targets >= start) & (message.targets < stop)
		used, sources = message.sources[mask].unique(sorted=True, return_inverse=True)
		return MessagePlan(
			message.value_nodes[used],
			sources,
			message.targets[mask] - start,
			None if message.weights is None else message.weights[mask],
		)

	return SparseLayerPlan(plan.targets[start:stop], select(plan.inputs), select(plan.edges))


def _ids(nodes: Tensor, count: int) -> None:
	if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
		raise ValueError("Sparse routing requires one-dimensional CPU int64 node IDs")
	if nodes.numel() and (nodes.min() < 0 or nodes.max() >= count):
		raise ValueError("Sparse routing node is outside the topology")


def _out_edges(graph: OutgoingTopology, nodes: Tensor, degrees: Tensor) -> tuple[Tensor, Tensor]:
	"""Assemble bounded routing metadata using individually bounded storage reads."""
	count = int(degrees.sum())
	if not count:
		empty = torch.empty(0, dtype=torch.int64)
		return empty, empty
	response_bytes = getattr(graph, "response_bytes", None)
	if response_bytes is None or (nodes.numel() + count) * 8 <= response_bytes:
		return graph.out_edges(nodes)
	if nodes.numel() == 1:
		raise PlanTooLarge("A single outgoing adjacency row exceeds the storage response budget")
	middle = nodes.numel() // 2
	left = _out_edges(graph, nodes[:middle], degrees[:middle])
	right = _out_edges(graph, nodes[middle:], degrees[middle:])
	return torch.cat((left[0], right[0])), torch.cat((left[1], right[1]))


def compile_delta_routes(
	graph: OutgoingTopology,
	changed_inputs: Tensor,
	edge_sources: Tensor,
	edge_targets: Tensor,
	edge_weights: Tensor,
	*,
	max_edges: int,
) -> SparseLayerPlan:
	"""Route input deltas over NEW outgoing rows and old inputs over edge changes.

	Self rows are included because GraphSAGE has a separate self transform. Edge
	weights encode additions/removals including parallel-edge multiplicity; a
	delete-pair request is not itself a multiplicity count. Source values are loaded
	separately at the appropriate old/new dependency versions by the caller.
	"""
	if type(max_edges) is not int or max_edges < 0:
		raise ValueError("Sparse routing requires a nonnegative edge metadata budget")
	count = graph.num_nodes()
	for nodes in (changed_inputs, edge_sources, edge_targets):
		_ids(nodes, count)
	if (
		edge_sources.shape != edge_targets.shape
		or edge_weights.shape != edge_sources.shape
		or edge_weights.dtype != torch.int64
		or edge_weights.device.type != "cpu"
	):
		raise ValueError("Structural routes require matching endpoints and signed int64 weights")
	changed = changed_inputs.unique(sorted=True)
	visible = edge_weights != 0
	edge_sources, edge_targets, edge_weights = edge_sources[visible], edge_targets[visible], edge_weights[visible]
	degrees = graph.out_degrees(changed)
	edge_count = int(degrees.sum())
	if edge_count + edge_sources.numel() > max_edges:
		raise PlanTooLarge("Sparse routing metadata exceeds the edge budget; use bounded recomputation")
	sources, destinations = _out_edges(graph, changed, degrees)
	if sources.numel() != edge_count or destinations.shape != sources.shape:
		raise RuntimeError("Topology changed while compiling sparse routing metadata")
	targets = torch.cat((changed, destinations, edge_targets)).unique(sorted=True)
	old_nodes = edge_sources.unique(sorted=True)
	return SparseLayerPlan(
		targets,
		MessagePlan(changed, torch.searchsorted(changed, sources), torch.searchsorted(targets, destinations), None),
		MessagePlan(
			old_nodes,
			torch.searchsorted(old_nodes, edge_sources),
			torch.searchsorted(targets, edge_targets),
			edge_weights,
		),
	)
