"""Sparse sum algebra; serving ownership/protocol integration remains separate.

For S=A H, use S' = S + A' (H'-H) + (A'-A) H. Routing the input
delta over the new topology includes the cross term for simultaneous edge and
feature changes. Edge corrections use old source inputs and signed multiplicity.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class RoutedRows:
	"""Compact source-value table and local source/target row indices.

	Parallel edges may remain repeated index pairs or be represented by integer
	weights. Keeping one value per source avoids repeating feature vectors in the
	communication body for every destination edge.
	"""

	values: Tensor
	sources: Tensor
	targets: Tensor
	weights: Optional[Tensor] = None

	def validate(self, sums: Tensor) -> None:
		if (
			self.values.ndim != 2
			or self.values.shape[1] != sums.shape[1]
			or self.values.device != sums.device
			or self.values.dtype not in (torch.float32, torch.float64)
			or self.sources.device != sums.device
			or self.targets.device != sums.device
			or self.sources.dtype != torch.int64
			or self.targets.dtype != torch.int64
			or self.sources.ndim != 1
			or self.targets.shape != self.sources.shape
		):
			raise ValueError("Routed deltas require aligned local indices and source feature rows")
		if self.sources.numel() and (
			self.sources.min() < 0
			or self.sources.max() >= self.values.shape[0]
			or self.targets.min() < 0
			or self.targets.max() >= sums.shape[0]
		):
			raise ValueError("Routed delta row is outside its source/target table")
		if self.weights is not None and (
			self.weights.device != sums.device
			or self.weights.dtype != torch.int64
			or self.weights.shape != self.sources.shape
		):
			raise ValueError("Routed multiplicities require one signed int64 weight per edge")


@dataclass(frozen=True)
class NeighborUpdate:
	sums: Tensor
	degrees: Tensor

	def mean(self, *, dtype: torch.dtype = torch.float32) -> Tensor:
		return (self.sums / self.degrees.clamp_min(1).unsqueeze(1)).to(dtype)


@torch.inference_mode()
def update_neighbor_sums(
	sums: Tensor, degrees: Tensor, input_delta: RoutedRows, edge_delta: RoutedRows, *, chunk_edges: int = 8192
) -> NeighborUpdate:
	"""Prepare replacement rows without modifying published inputs.

	Only the selected aggregate rows are copied. Per-edge vector scratch is
	bounded by chunk_edges * width, independently of the routed edge count.
	This primitive does not allocate or retain a whole-graph embedding snapshot.
	"""
	if (
		sums.ndim != 2
		or sums.dtype not in (torch.float32, torch.float64)
		or degrees.dtype != torch.int64
		or degrees.device != sums.device
		or degrees.shape != (sums.shape[0],)
		or (degrees < 0).any()
		or type(chunk_edges) is not int
		or chunk_edges < 1
	):
		raise ValueError("Invalid aggregate rows, degrees or edge scratch bound")
	input_delta.validate(sums)
	edge_delta.validate(sums)
	if edge_delta.weights is None:
		raise ValueError("Structural corrections require explicit signed edge multiplicities")
	updated_degrees = degrees.clone()
	updated_degrees.index_add_(0, edge_delta.targets, edge_delta.weights)
	if (updated_degrees < 0).any():
		raise ValueError("Structural correction would create a negative degree")
	updated = sums.clone()
	for message in (input_delta, edge_delta):
		for start in range(0, message.sources.numel(), chunk_edges):
			end = start + chunk_edges
			values = message.values[message.sources[start:end]].to(sums.dtype)
			if message.weights is not None:
				values.mul_(message.weights[start:end].unsqueeze(1))
			updated.index_add_(0, message.targets[start:end], values)
	# Empty neighborhoods are exactly zero, including after complete deletion;
	# do not keep residual cancellation error in isolated-node state.
	updated[updated_degrees == 0] = 0
	return NeighborUpdate(updated, updated_degrees)
