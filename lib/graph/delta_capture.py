"""Bounded old-input capture before authoritative feature/topology mutation.

The caller holds the graph/version fence across capture and mutation. These
records do not mutate storage or certify a worker aggregate's dependency anchor.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from lib.graph.recompute import PlanTooLarge


class Topology(Protocol):
	def num_nodes(self) -> int: ...


@dataclass(frozen=True)
class FeatureSnapshot:
	nodes: Tensor
	values: Tensor


@dataclass(frozen=True)
class EdgeCorrections:
	sources: Tensor
	targets: Tensor
	weights: Tensor


def _ids(nodes: Tensor, count: int) -> None:
	if nodes.device.type != "cpu" or nodes.dtype != torch.int64 or nodes.ndim != 1:
		raise ValueError("Delta capture requires CPU int64 node vectors")
	if nodes.numel() and (nodes.min() < 0 or nodes.max() >= count):
		raise ValueError("Delta capture node is outside storage")


def capture_features(features: Tensor, nodes: Tensor, *, max_bytes: int) -> FeatureSnapshot:
	"""Copy only selected old rows; in-place mutation makes this copy necessary."""
	if features.device.type != "cpu" or features.ndim != 2 or not features.is_floating_point():
		raise ValueError("Feature snapshots require a CPU floating-point row table")
	_ids(nodes, features.shape[0])
	selected = nodes.unique(sorted=True)
	if type(max_bytes) is not int or max_bytes < 0:
		raise ValueError("Old-feature capture requires a nonnegative byte budget")
	if selected.numel() * (8 + features.shape[1] * features.element_size()) > max_bytes:
		raise PlanTooLarge("Old-feature snapshot exceeds its byte budget; use bounded recomputation")
	return FeatureSnapshot(selected, torch.index_select(features, 0, selected))


def _multiplicities(graph: Topology, sources: Tensor, targets: Tensor) -> Tensor:
	if not sources.numel():
		return torch.zeros_like(sources)
	count = getattr(graph, "edge_multiplicities", None)
	if count is not None:
		return cast(Tensor, count(sources, targets))
	# DGL's borrowed CSR avoids materializing matching edge IDs or expanding a
	# feature vector per parallel edge. Scan one source row once for all queries.
	csr = getattr(graph, "adj_tensors", None)
	if csr is None:
		raise TypeError("Topology must expose multiplicities or borrowed CPU CSR rows")
	offsets, neighbors, _ = csr("csr")
	if offsets.device.type != "cpu" or neighbors.device.type != "cpu":
		raise ValueError("Delta topology capture belongs to CPU storage")
	result = torch.zeros_like(sources)
	for source in sources.unique(sorted=True).tolist():
		positions = (sources == source).nonzero().flatten()
		wanted, order = targets[positions].sort()
		total = torch.zeros_like(wanted)
		row = neighbors[int(offsets[source]) : int(offsets[source + 1])]
		for chunk in row.split(65536):
			indices = torch.searchsorted(wanted, chunk)
			matches = (indices < wanted.numel()) & (wanted[indices.clamp_max(wanted.numel() - 1)] == chunk)
			total += torch.bincount(indices[matches], minlength=wanted.numel())
		result[positions[order]] = total
	return result


def capture_edges(graph: Topology, sources: Tensor, targets: Tensor, *, delete: bool) -> EdgeCorrections:
	"""Coalesce actual signed multiplicities before insertion/delete-all-pairs."""
	_ids(sources, graph.num_nodes())
	_ids(targets, graph.num_nodes())
	if sources.shape != targets.shape or type(delete) is not bool:
		raise ValueError("Delta edge capture requires aligned endpoints and a boolean deletion mode")
	pairs, counts = torch.unique(torch.stack((sources, targets), dim=1), dim=0, return_counts=True)
	first, second = pairs[:, 0].contiguous(), pairs[:, 1].contiguous()
	weights = -_multiplicities(graph, first, second) if delete else counts
	if weights.shape != first.shape or weights.dtype != torch.int64 or weights.device.type != "cpu":
		raise ValueError("Storage multiplicities must be aligned CPU int64 counts")
	if delete and (weights > 0).any():
		raise ValueError("Storage reported a negative edge multiplicity")
	visible = weights != 0
	return EdgeCorrections(first[visible], second[visible], weights[visible])
