"""Layered GNN model and aggregate-to-output transforms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock

from lib.models.registry import ADAPTED_MODELS, MODEL_LAYERS


class GNNModel(nn.Module):
	def __init__(
		self, model_name: str, num_features: int, num_hiddens: int, num_classes: int, num_layers: int, **kwargs
	):
		super().__init__()

		self.model_name = model_name

		self.num_layers = num_layers
		self.num_features = num_features
		self.num_hiddens = num_hiddens
		self.num_classes = num_classes

		model_kwargs = {}
		if model_name == "gat":
			model_kwargs["num_heads"] = kwargs.get("num_heads", 1)
		elif model_name == "sage":
			model_kwargs["aggregator_type"] = kwargs.get("aggregator_type", "mean")

		if model_name in ADAPTED_MODELS:
			model_kwargs = {key: kwargs[key] for key in ("relations", "pseudo_dim", "kernels") if key in kwargs}
		self.layers = nn.ModuleList()
		cls = MODEL_LAYERS.get(model_name)
		if cls is None:
			raise ValueError(f"Unknown GNN model: {model_name}")
		for i in range(num_layers):
			in_hidden = num_features if i == 0 else num_hiddens
			out_hidden = num_classes if i == num_layers - 1 else num_hiddens
			self.layers.append(cls(in_hidden, out_hidden, **model_kwargs))

	def from_aggrs(self, layer_idx, self_features, neighbor_mean, *, projected=False):
		"""Mean GraphSAGE transform; projected means W_neigh is already applied."""
		layer = self.layers[layer_idx]
		if self.training:
			raise ValueError("Aggregate execution requires model.eval()")
		if self.model_name != "sage" or layer._aggre_type != "mean":
			raise NotImplementedError("Aggregate execution currently supports mean GraphSAGE")
		if projected and layer.fc_neigh.bias is not None:
			raise NotImplementedError("Projected sums require a bias-free neighbor transform")
		h_neigh = neighbor_mean if projected else layer.fc_neigh(neighbor_mean)
		h = layer.fc_self(self_features) + h_neigh
		if layer.activation is not None:
			h = layer.activation(h)
		if layer.norm is not None:
			h = layer.norm(h)
		return F.relu(h) if layer_idx + 1 < self.num_layers else h

	def forward(self, layer_idx: int, block: DGLBlock, h: torch.Tensor) -> torch.Tensor:
		layer = self.layers[layer_idx]
		h = layer(block, h)

		if self.model_name == "gat":
			h = h.mean(1)
		if layer_idx != self.num_layers - 1:
			h = F.relu(h)

		return h
