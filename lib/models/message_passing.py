"""Message, neighborhood-context, and update operators for GNN convolutions.

Operator decomposition reference: https://arxiv.org/abs/2603.20622 (Table II).

Optional edge data ``nrt`` has columns [relation, importance, pseudo...].
Relations are normalized independently. Missing metadata means relation zero,
unit importance and zero pseudo coordinates. Parameters stay fixed in eval mode.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MessagePassingConv(nn.Module):
	def __init__(self, name, in_feats, out_feats, *, relations=4, pseudo_dim=2, kernels=3):
		super().__init__()
		self.name, self.in_feats, self.out_feats = name, in_feats, out_feats
		self.relations = relations if name in ("rgcn", "rgat") else 1
		self.pseudo_dim = pseudo_dim
		initializers = []
		if name in ("pinsage", "ggcn", "agnn"):
			self.fc = nn.Linear(in_feats, out_feats, bias=False)
		if name in ("pinsage", "rgcn"):
			self.fc_self = nn.Linear(in_feats, out_feats)
		if name == "ggcn":
			self.source_gate = nn.Linear(in_feats, in_feats)
			self.destination_gate = nn.Linear(in_feats, in_feats, bias=False)
		elif name == "pinsage":
			self.project = nn.Linear(in_feats, in_feats)
		elif name in ("rgcn", "rgat"):
			self.rel_weight = nn.Parameter(torch.empty(self.relations, in_feats, out_feats))
			initializers.append(self.rel_weight)
			if name == "rgat":
				self.attn_l = nn.Parameter(torch.empty(self.relations, out_feats))
				self.attn_r = nn.Parameter(torch.empty(self.relations, out_feats))
				initializers.extend((self.attn_l, self.attn_r))
		elif name == "agnn":
			self.beta = nn.Parameter(torch.ones(()))
		elif name == "monet":
			self.mu = nn.Parameter(torch.zeros(kernels, pseudo_dim))
			self.inv_sigma = nn.Parameter(torch.ones(kernels, pseudo_dim))
			self.kernel_weight = nn.Parameter(torch.empty(kernels, in_feats, out_feats))
			initializers.append(self.kernel_weight)
		for parameter in initializers:
			nn.init.xavier_uniform_(parameter)

	@property
	def constrained(self):
		return self.name in ("rgat", "ggcn", "agnn")

	def attention_inputs(self, x, self_x, attributes):
		relations = (
			attributes[:, 0].long()
			if attributes.shape[1]
			else torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
		)
		if self.name == "rgat":
			values = torch.einsum("bi,bio->bo", x, self.rel_weight[relations])
			destination = (
				torch.einsum("...i,bio->bo", self_x, self.rel_weight[relations])
				if self_x.ndim == 1
				else torch.einsum("bi,bio->bo", self_x, self.rel_weight[relations])
			)
			logits = F.leaky_relu(
				(values * self.attn_l[relations]).sum(-1) + (destination * self.attn_r[relations]).sum(-1), 0.2
			)
		else:
			values = self.fc(x)
			logits = self.beta * (F.normalize(x, dim=-1) * F.normalize(self_x, dim=-1)).sum(-1)
		return values, logits.double(), relations

	def attention_shift(self, x, self_x, attributes):
		_, logits, relations = self.attention_inputs(x, self_x, attributes)
		shift = x.new_zeros(self.relations, dtype=torch.float64)
		for relation in range(self.relations):
			selected = logits[relations == relation]
			if selected.numel():
				shift[relation] = selected.max()
		return shift

	def messages(self, x, self_x, attributes, shift=None, *, edge_shift=None):
		count = x.shape[0]
		relations = (
			attributes[:, 0].long() if attributes.shape[1] else torch.zeros(count, device=x.device, dtype=torch.int64)
		)
		if attributes.shape[1] and (
			not torch.equal(attributes[:, 0], relations.to(attributes.dtype))
			or (relations < 0).any()
			or (relations >= self.relations).any()
		):
			raise ValueError("Edge relation is outside the configured model")
		importance = attributes[:, 1:2] if attributes.shape[1] > 1 else x.new_ones((count, 1))
		context = x.new_ones((count, 1), dtype=torch.float64)
		if self.name == "pinsage":
			values = self.fc(F.relu(self.project(x))) * importance
		elif self.name == "rgcn":
			values = torch.einsum("bi,bio->bo", x, self.rel_weight[relations])
		elif self.name in ("rgat", "agnn"):
			values, logits, _ = self.attention_inputs(x, self_x, attributes)
			offset = edge_shift if edge_shift is not None else shift[relations] if shift is not None else 0
			context = (logits - offset).exp().unsqueeze(-1)
			values = values.double() * context
		elif self.name == "ggcn":
			values = self.fc(x * torch.sigmoid(self.source_gate(x) + self.destination_gate(self_x)))
		elif self.name == "monet":
			pseudo = attributes[:, 2:] if attributes.shape[1] > 2 else x.new_zeros((count, self.pseudo_dim))
			if pseudo.shape[1] != self.pseudo_dim:
				raise ValueError("MoNet pseudo coordinate width differs from the model")
			gaussian = (-0.5 * ((pseudo[:, None, :] - self.mu) * self.inv_sigma).square().sum(-1)).exp()
			values = (torch.einsum("bi,kio->bko", x, self.kernel_weight) * gaussian.unsqueeze(-1)).sum(1)
		else:
			raise ValueError(f"Unsupported message-passing model: {self.name}")
		selector = F.one_hot(relations, self.relations).to(values.dtype)
		return values.double().unsqueeze(1) * selector.unsqueeze(-1), context.unsqueeze(1) * selector.unsqueeze(-1)

	def finish(self, self_x, numerator, context):
		if self.name in ("pinsage", "rgcn", "rgat", "agnn"):
			numerator = numerator / context.clamp_min(torch.finfo(torch.float64).tiny)
		value = numerator.sum(-2).to(self_x.dtype)
		return value + self.fc_self(self_x) if self.name in ("pinsage", "rgcn") else value

	def forward(self, graph, inputs):
		u, v = graph.edges()
		attributes = graph.edata.get("nrt", inputs.new_empty((u.numel(), 0)))
		x, self_x = inputs[u], inputs[v]
		shift = None
		if self.name in ("rgat", "agnn"):
			_, logits, relations = self.attention_inputs(x, self_x, attributes)
			groups = v * self.relations + relations
			maximum = logits.new_full((graph.num_dst_nodes() * self.relations,), -torch.inf)
			maximum.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
			shift = maximum[groups]
		messages, context = self.messages(x, self_x, attributes, edge_shift=shift)
		sums = messages.new_zeros((graph.num_dst_nodes(), self.relations, self.out_feats))
		contexts = context.new_zeros((graph.num_dst_nodes(), self.relations, 1))
		sums.index_add_(0, v, messages)
		contexts.index_add_(0, v, context)
		return self.finish(inputs[: graph.num_dst_nodes()], sums, contexts)
