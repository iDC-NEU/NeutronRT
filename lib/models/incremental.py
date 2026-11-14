"""Signed replacement aggregation (arXiv:2603.20622, Algorithms 1 and 3).

Historical state is a bounded CPU optimization, never authoritative. Every task
still supplies a complete, version-certified neighborhood. Eviction, migration,
or process replacement therefore needs no cache coordination or wire changes.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lib.graph.recompute import LayerPlan
from lib.models import MODEL_NAMES, GNNModel
from lib.models.message_passing import MessagePassingConv
from lib.models.operators import full_neighbor_forward


@dataclass
class RowState:
	sources: Tensor
	inputs: Tensor
	degrees: Tensor
	counts: Tensor
	attributes: Tensor
	self_input: Tensor
	numerator: Tensor
	context: Tensor
	shift: Tensor
	steps: int

	@property
	def nbytes(self) -> int:
		return sum(v.numel() * v.element_size() for v in vars(self).values() if isinstance(v, Tensor))


class IncrementalLayers:
	"""Apply native model transforms to incrementally maintained neighborhood state.

	FP64 accumulation limits cancellation. Periodic full refresh bounds correction
	chains. Destination-dependent messages refresh on changed self input, exactly
	as required by the constrained-model rule in Section IV-C. Model weights must
	remain fixed for this object's lifetime (inference only).
	"""

	def __init__(
		self,
		model: GNNModel,
		*,
		max_bytes: int = 256 * 1024**2,
		refresh_steps: int = 32,
		mapped_host_reads: bool = False,
	):
		if model.training or model.model_name not in MODEL_NAMES:
			raise ValueError("Incremental layers require a supported fixed eval-mode GNN")
		if max_bytes < 0 or refresh_steps < 1:
			raise ValueError("Invalid incremental state budget or refresh interval")
		self.model, self.max_bytes, self.refresh_steps = model, max_bytes, refresh_steps
		self.mapped_host_reads = mapped_host_reads
		self.rows: OrderedDict[tuple[int, int], RowState] = OrderedDict()
		self.bytes_used = 0
		self.messages = self.reused = self.refreshes = 0

	def clear(self):
		self.rows.clear()
		self.bytes_used = 0

	def _device_rows(self, values, selected, device):
		if not self.mapped_host_reads or device.type != "cuda" or not values.numel() or not selected.numel():
			return values[selected].to(device)
		from lib.storage.pinned import PinnedHostRows

		with PinnedHostRows(values) as rows:
			result = rows.take(selected.to(device))
			# Registration must outlive the gather kernel before it is released.
			torch.cuda.current_stream(device).synchronize()
			return result

	def _put(self, key, state):
		previous = self.rows.pop(key, None)
		if previous is not None:
			self.bytes_used -= previous.nbytes
		if state.nbytes > self.max_bytes:
			return
		while self.rows and self.bytes_used + state.nbytes > self.max_bytes:
			_, old = self.rows.popitem(last=False)
			self.bytes_used -= old.nbytes
		self.rows[key] = state
		self.bytes_used += state.nbytes

	def _messages(self, layer, x, self_x, degrees, shift, attributes):
		name = self.model.model_name
		self.messages += x.shape[0]
		if isinstance(layer, MessagePassingConv):
			return layer.messages(x, self_x, attributes, shift if layer.name in ("rgat", "agnn") else None)
		if name == "gat":
			projected = layer.fc(x).view(-1, layer._num_heads, layer._out_feats)
			destination = layer.fc(self_x).view(layer._num_heads, layer._out_feats)
			logits = layer.leaky_relu(
				(projected * layer.attn_l).sum(-1) + (destination * layer.attn_r).sum(-1)
			).double()
			scores = (logits - shift).exp().unsqueeze(-1)
			return projected.double() * scores, scores
		if name == "gcn":
			power = -0.5 if layer._norm == "both" else -1.0
			values = x.double()
			if layer._norm in ("left", "both"):
				values = values * degrees.clamp_min(1).double().pow(power).unsqueeze(-1)
		else:
			values = x.double()
		return values, x.new_ones((x.shape[0], 1), dtype=torch.float64)

	def _finish(self, layer, self_x, numerator, context, count):
		name = self.model.model_name
		if isinstance(layer, MessagePassingConv):
			return layer.finish(self_x, numerator, context)
		if name == "gat":
			value = (numerator / context.clamp_min(torch.finfo(torch.float64).tiny)).to(self_x.dtype)
			if layer.res_fc is not None:
				value = value + layer.res_fc(self_x).view(-1, layer._out_feats)
			if layer.has_explicit_bias:
				value = value + layer.bias.view(layer._num_heads, layer._out_feats)
			if layer.activation is not None:
				value = layer.activation(value)
			return value.mean(0)
		value = numerator.to(self_x.dtype)
		if name in ("sage", "commnet"):
			if name == "sage":
				value = value / max(count, 1)
			value = layer.fc_self(self_x) + layer.fc_neigh(value)
			if layer.activation is not None:
				value = layer.activation(value)
			if layer.norm is not None:
				value = layer.norm(value.unsqueeze(0)).squeeze(0)
		elif name == "gin":
			value = value + (1 + layer.eps) * self_x
			if layer.apply_func is not None:
				value = layer.apply_func(value)
			if layer.activation is not None:
				value = layer.activation(value)
		else:
			if layer.weight is not None:
				value = value @ layer.weight
			if layer._norm in ("right", "both"):
				value = value * max(count, 1) ** (-0.5 if layer._norm == "both" else -1.0)
			if layer.bias is not None:
				value = value + layer.bias
			if layer._activation is not None:
				value = layer._activation(value)
		return value

	def _initialize_sage(self, task, inputs, source_ids, degrees, u, edge_order, row_offsets) -> Tensor:
		"""Batch cold forward math; retain CPU snapshots without per-row CUDA launches."""
		values = full_neighbor_forward(self.model.layers[task.layer - 1], task, inputs)
		self.messages += u.numel()
		self.refreshes += task.targets.numel()
		if self.max_bytes:
			cpu_inputs = inputs.cpu()
			for row, target in enumerate(task.targets.tolist()):
				selected = edge_order[row_offsets[row] : row_offsets[row + 1]]
				# Preserve first-occurrence ordering and parallel-edge multiplicities.
				index_array, first, count_array = np.unique(u[selected].numpy(), return_index=True, return_counts=True)
				order = np.argsort(first)
				indices, counts = torch.from_numpy(index_array[order]), torch.from_numpy(count_array[order])
				x = cpu_inputs[indices]
				self._put(
					(task.layer, target),
					RowState(
						source_ids[indices],
						x,
						degrees[indices],
						counts,
						x.new_empty((indices.numel(), 0)),
						cpu_inputs[row].clone(),
						(x.double() * counts[:, None]).sum(0),
						torch.tensor([selected.numel()], dtype=torch.float64),
						x.new_zeros(()),
						0,
					),
				)
		return values.relu() if task.layer < self.model.num_layers else values

	@torch.inference_mode()
	def apply(self, task: LayerPlan, inputs: Tensor) -> Tensor:
		layer = self.model.layers[task.layer - 1]
		if self.model.model_name == "sage" and layer._aggre_type != "mean":
			raise ValueError("Incremental GraphSAGE requires mean aggregation")
		if inputs.ndim != 2 or inputs.shape[0] != task.sources.numel():
			raise ValueError("Inputs do not match compiled source rows")
		if not torch.equal(task.sources[: task.targets.numel()], task.targets):
			raise ValueError("Destination self rows must prefix task sources")
		u, v = (edge.cpu() for edge in task.block.edges())
		source_ids = task.sources.cpu()
		attributes = task.block.edata.get("nrt", inputs.new_empty((u.numel(), 0))).cpu()
		degrees = task.source_out_degrees
		if self.model.model_name == "gcn" and degrees is None:
			raise ValueError("GCN incremental tasks require full-graph source degrees")
		degrees = degrees.cpu() if degrees is not None else torch.zeros(source_ids.numel(), dtype=torch.int64)
		# One stable grouping replaces an O(targets * edges) scan in every task.
		edge_order = v.argsort(stable=True)
		row_offsets_tensor = torch.cat(
			[torch.zeros(1, dtype=torch.int64), v.bincount(minlength=task.targets.numel()).cumsum(0)]
		)
		row_offsets = row_offsets_tensor.tolist()
		if (
			self.model.model_name == "sage"
			and attributes.shape[1] == 0
			and all((task.layer, target) not in self.rows for target in task.targets.tolist())
		):
			return self._initialize_sage(task, inputs, source_ids, degrees, u, edge_order, row_offsets)
		cpu_inputs = inputs.cpu()
		outputs = []
		for row, target in enumerate(task.targets.cpu().tolist()):
			selected = edge_order[row_offsets[row] : row_offsets[row + 1]]
			groups = {}
			for edge in selected.tolist():
				signature = (int(source_ids[u[edge]]), *attributes[edge].tolist())
				if signature not in groups:
					groups[signature] = [int(u[edge]), edge, 0]
				groups[signature][2] += 1
			entries = list(groups.values())
			indices = torch.tensor([entry[0] for entry in entries], dtype=torch.int64)
			counts = torch.tensor([entry[2] for entry in entries], dtype=torch.int64)
			attrs = attributes[torch.tensor([entry[1] for entry in entries], dtype=torch.int64)]
			ids = source_ids[indices]
			x, deg = cpu_inputs[indices], degrees[indices]
			self_x = cpu_inputs[row]
			key = task.layer, target
			old = self.rows.get(key)
			refresh = old is None or old.steps >= self.refresh_steps
			constrained = self.model.model_name == "gat" or isinstance(layer, MessagePassingConv) and layer.constrained
			if old is not None and constrained and not torch.equal(self_x, old.self_input):
				refresh = True
			if old is not None and old.attributes.shape[1] != attrs.shape[1]:
				refresh = True
			shift = (
				inputs.new_zeros((layer._num_heads,), dtype=torch.float64)
				if self.model.model_name == "gat"
				else inputs.new_zeros(())
			)
			if not refresh:
				assert old is not None
				shift = old.shift.to(inputs.device)
				old_keys = [(int(node), *attr) for node, attr in zip(old.sources, old.attributes.tolist(), strict=True)]
				new_keys = [(int(node), *attr) for node, attr in zip(ids, attrs.tolist(), strict=True)]
				old_map, new_map = (
					{key: i for i, key in enumerate(old_keys)},
					{key: i for i, key in enumerate(new_keys)},
				)
				union = list(dict.fromkeys([*old_keys, *new_keys]))
				old_index = torch.tensor([old_map.get(key, -1) for key in union], dtype=torch.int64)
				new_index = torch.tensor([new_map.get(key, -1) for key in union], dtype=torch.int64)
				both = (old_index >= 0) & (new_index >= 0)
				same = torch.zeros_like(both)
				oi, ni = old_index[both], new_index[both]
				same[both] = (
					(old.inputs[oi] == x[ni]).all(1) & (old.degrees[oi] == deg[ni]) & (old.counts[oi] == counts[ni])
				)
				remove = old_index[(old_index >= 0) & ~same]
				add = new_index[(new_index >= 0) & ~same]
				self.reused += int(same.sum())
				numerator, context = old.numerator.to(inputs.device), old.context.to(inputs.device)
				for sign, selected, values, ds, multiplicity, edge_attrs in (
					(-1, remove, old.inputs, old.degrees, old.counts, old.attributes),
					(1, add, x, deg, counts, attrs),
				):
					msg, ctx = self._messages(
						layer,
						self._device_rows(values, selected, inputs.device),
						self_x.to(inputs.device),
						ds[selected].to(inputs.device),
						shift,
						edge_attrs[selected].to(inputs.device),
					)
					weights = multiplicity[selected].to(inputs.device).reshape((-1,) + (1,) * (msg.ndim - 1))
					numerator = numerator + sign * (msg * weights).sum(0)
					context = context + sign * (ctx * weights).sum(0)
				# Cancellation or a score beyond the retained exponential scale needs
				# a stable fresh reduction, not clamping an invalid attention sum.
				active = torch.ones_like(context, dtype=torch.bool)
				if isinstance(layer, MessagePassingConv):
					relations = attrs[:, 0].long() if attrs.shape[1] else torch.zeros(ids.numel(), dtype=torch.int64)
					active = (torch.bincount(relations, minlength=layer.relations) > 0).to(inputs.device).unsqueeze(-1)
					numerator = torch.where(active, numerator, 0)
					context = torch.where(active, context, 0)
				if (
					not torch.isfinite(numerator).all()
					or not torch.isfinite(context).all()
					or (
						old.numerator.abs().max() > 0
						and numerator.abs().max()
						<= old.numerator.abs().max().to(inputs.device) * (32 * torch.finfo(inputs.dtype).eps)
					)
					or (counts.sum() > 0 and ((context <= 0) & active).any())
					or constrained
					and ((context < old.context.to(inputs.device) * 1e-10) & active).any()
				):
					refresh = True
			if refresh:
				self.refreshes += 1
				if isinstance(layer, MessagePassingConv) and layer.name in ("rgat", "agnn"):
					shift = layer.attention_shift(
						x.to(inputs.device), self_x.to(inputs.device), attrs.to(inputs.device)
					)
				if self.model.model_name == "gat" and x.numel():
					projected = layer.fc(x.to(inputs.device)).view(-1, layer._num_heads, layer._out_feats)
					destination = layer.fc(self_x.to(inputs.device)).view(layer._num_heads, layer._out_feats)
					shift = (
						layer.leaky_relu((projected * layer.attn_l).sum(-1) + (destination * layer.attn_r).sum(-1))
						.double()
						.amax(0)
					)
				msg, ctx = self._messages(
					layer,
					x.to(inputs.device),
					self_x.to(inputs.device),
					deg.to(inputs.device),
					shift,
					attrs.to(inputs.device),
				)
				weights = counts.to(inputs.device).reshape((-1,) + (1,) * (msg.ndim - 1))
				numerator, context = (msg * weights).sum(0), (ctx * weights).sum(0)
			if not counts.numel():
				numerator.zero_()
				context.zero_()
			value = self._finish(layer, self_x.to(inputs.device), numerator, context, int(counts.sum()))
			outputs.append(value.relu() if task.layer < self.model.num_layers else value)
			self._put(
				key,
				RowState(
					ids.clone(),
					x.clone(),
					deg.clone(),
					counts.clone(),
					attrs.clone(),
					self_x.clone(),
					numerator.cpu(),
					context.cpu(),
					shift.cpu(),
					0 if refresh or old is None else old.steps + 1,
				),
			)
		width = self.model.num_classes if task.layer == self.model.num_layers else self.model.num_hiddens
		return torch.stack(outputs) if outputs else inputs.new_empty((0, width))
