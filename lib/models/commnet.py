"""CommNet self transform plus summed neighbor messages."""

import dgl.function as fn
from torch import nn


class CommNetConv(nn.Module):
	def __init__(self, in_feats, out_feats):
		super().__init__()
		self.fc_self = nn.Linear(in_feats, out_feats)
		self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
		self.activation = self.norm = None

	def forward(self, graph, features):
		with graph.local_scope():
			graph.srcdata["h"] = features
			graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "a"))
			return self.fc_self(features[: graph.num_dst_nodes()]) + self.fc_neigh(graph.dstdata["a"])
