"""Reconstruct selected mean-GraphSAGE rows without retaining layer embeddings."""

from dataclasses import replace
from typing import Sequence, cast

import dgl.function as fn
import torch
from torch import Tensor, nn

from lib.graph.deltas import NeighborUpdate, RoutedRows, update_neighbor_sums
from lib.graph.recompute import LayerPlan
from lib.models import GNNModel


def delta_sage_workspace_bytes(
	targets: int,
	changed: int,
	structural: int,
	input_width: int,
	output_width: int,
	routed_edges: int,
	*,
	chunk_edges: int = 8192,
) -> int:
	"""Bound outputs, differences, projections, mean/linear temporaries and scatter.

	This excludes input tables and their wire storage. The five target-width
	matrices include the two replacement outputs and transient mean/linear/ReLU
	results; projections and differences are conservatively counted together.
	"""
	return (
		4
		* (
			5 * targets * output_width
			+ changed * input_width
			+ (changed + structural) * output_width
			+ min(routed_edges, chunk_edges) * output_width
		)
		+ 40 * targets
	)


@torch.inference_mode()
def full_sage_state(model: GNNModel, task: LayerPlan, inputs: Tensor) -> tuple[Tensor, Tensor]:
	"""Initialize a full-neighbor projected sum and h without a second edge pass.

	Storage compiles all incoming edges and puts destination self rows first in
	the source table. The returned tensors are producer-local replacement state;
	publication and their lifetime are the caller's responsibility.
	"""
	if model.training or model.model_name != "sage" or not 1 <= task.layer <= model.num_layers:
		raise ValueError("Aggregate initialization requires an eval-mode GraphSAGE layer")
	layer = model.layers[task.layer - 1]
	if getattr(layer, "_aggre_type", None) != "mean":
		raise ValueError("Aggregate initialization requires mean GraphSAGE")
	neighbor = cast(nn.Linear, layer.fc_neigh)
	if neighbor.bias is not None:
		raise ValueError("Projected sums require a bias-free neighbor transform")
	if inputs.shape != (task.sources.numel(), neighbor.in_features) or inputs.device != task.block.device:
		raise ValueError("Inputs must follow the full-neighbor source table on its device")
	if task.targets.numel() > task.sources.numel() or not torch.equal(
		task.sources[: task.targets.numel()], task.targets
	):
		raise ValueError("Destination self rows must prefix aggregate sources")
	with task.block.local_scope():
		if task.block.num_edges():
			task.block.srcdata["_nrt_projected"] = neighbor(inputs)
			task.block.update_all(fn.copy_u("_nrt_projected", "_nrt_message"), fn.sum("_nrt_message", "_nrt_sum"))
			sums = cast(Tensor, task.block.dstdata["_nrt_sum"])
		else:
			sums = inputs.new_zeros((task.targets.numel(), neighbor.out_features))
		mean = sums / task.block.in_degrees().clamp_min(1).unsqueeze(1)
		output = model.from_aggrs(task.layer - 1, inputs[: task.targets.numel()], mean, projected=True)
	return output, sums


@torch.inference_mode()
def reconstruct_sage(
	model: GNNModel,
	features: Tensor,
	aggregates: Sequence[NeighborUpdate],
	*,
	projected: bool = False,
	start_layer: int = 0,
) -> Tensor:
	"""Rebuild h^L(v) from x(v), a^1(v), ..., a^L(v), with no neighbor traversal.

	All tables must follow the same selected node order and a consistent dependency
	version. The caller certifies freshness and model identity before this operation;
	array shape is not evidence of either. Temporary lower-layer embeddings are
	computed for these selected nodes only and are not cached here.
	"""
	if (
		model.training
		or model.model_name != "sage"
		or any(getattr(layer, "_aggre_type", None) != "mean" for layer in model.layers)
	):
		raise ValueError("Aggregate-only reconstruction requires eval-mode mean GraphSAGE")
	if type(start_layer) is not int or not 0 <= start_layer < model.num_layers:
		raise ValueError("Reconstruction start layer is outside the model")
	if not 1 <= len(aggregates) <= model.num_layers - start_layer:
		raise ValueError("Supply aggregate rows through the requested model layer")
	self_transform = cast(nn.Linear, model.layers[start_layer].fc_self)
	first = self_transform.weight
	if (
		features.ndim != 2
		or features.shape[1] != self_transform.in_features
		or features.dtype != first.dtype
		or features.device != first.device
	):
		raise ValueError("Selected features must match the model's input width, dtype and device")
	for index, state in enumerate(aggregates, start=start_layer):
		neighbor = cast(nn.Linear, model.layers[index].fc_neigh)
		width = neighbor.out_features if projected else neighbor.in_features
		if projected and neighbor.bias is not None:
			raise ValueError("Projected sums require a bias-free neighbor transform")
		if (
			state.sums.shape != (features.shape[0], width)
			or state.sums.device != features.device
			or state.sums.dtype not in (torch.float32, torch.float64)
			or state.degrees.shape != (features.shape[0],)
			or state.degrees.device != features.device
			or state.degrees.dtype != torch.int64
			or (state.degrees < 0).any()
		):
			raise ValueError("Aggregate rows must match selected nodes, layer width, device and nonnegative degrees")
	current = features
	for index, state in enumerate(aggregates, start=start_layer):
		current = model.from_aggrs(index, current, state.mean(dtype=first.dtype), projected=projected)
	return cast(Tensor, current)


@torch.inference_mode()
def delta_sage_state(
	model: GNNModel,
	layer: int,
	self_features: Tensor,
	state: NeighborUpdate,
	input_delta: RoutedRows,
	edge_delta: RoutedRows,
	*,
	chunk_edges: int = 8192,
) -> tuple[Tensor, NeighborUpdate]:
	"""Update selected projected sums and h at a one-based model layer.

	The caller certifies the old aggregate/degree anchor. Input deltas route over
	the new adjacency; structural corrections carry exact old input values. Only
	compact source tables are projected, followed by bounded edge scatter scratch.
	"""
	if model.training or model.model_name != "sage" or not 1 <= layer <= model.num_layers:
		raise ValueError("Sparse aggregate updates require an eval-mode GraphSAGE layer")
	operator = model.layers[layer - 1]
	neighbor = cast(nn.Linear, operator.fc_neigh)
	if getattr(operator, "_aggre_type", None) != "mean" or neighbor.bias is not None:
		raise ValueError("Sparse projected sums require mean GraphSAGE and bias-free neighbor weights")
	if (
		state.sums.ndim != 2
		or state.sums.shape[1] != neighbor.out_features
		or self_features.shape != (state.sums.shape[0], neighbor.in_features)
		or self_features.device != neighbor.weight.device
		or self_features.dtype != neighbor.weight.dtype
	):
		raise ValueError("Sparse self inputs and projected sums must follow selected target rows")
	for message in (input_delta, edge_delta):
		if (
			message.values.ndim != 2
			or message.values.shape[1] != neighbor.in_features
			or message.values.dtype != neighbor.weight.dtype
			or message.values.device != neighbor.weight.device
		):
			raise ValueError("Sparse source tables must match the layer input width, device and dtype")
	updated = update_neighbor_sums(
		state.sums,
		state.degrees,
		replace(input_delta, values=neighbor(input_delta.values)),
		replace(edge_delta, values=neighbor(edge_delta.values)),
		chunk_edges=chunk_edges,
	)
	output = model.from_aggrs(layer - 1, self_features, updated.mean(dtype=neighbor.weight.dtype), projected=True)
	return output, updated
