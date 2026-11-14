"""Supported GNN layers; each name has one implementation."""

from functools import partial

import dgl.nn as dglnn
import torch.nn as nn

from lib.models.commnet import CommNetConv
from lib.models.message_passing import MessagePassingConv

ADAPTED_MODELS = ("pinsage", "rgcn", "rgat", "ggcn", "agnn", "monet")

MODEL_LAYERS = {
	"commnet": CommNetConv,
	"gcn": dglnn.GraphConv,
	"sage": dglnn.SAGEConv,
	"gat": dglnn.GATConv,
}


def _gin_layer(in_feats, out_feats):
	return dglnn.GINConv(nn.Linear(in_feats, out_feats), "sum", learn_eps=True)


MODEL_LAYERS.update({name: partial(MessagePassingConv, name) for name in ADAPTED_MODELS})
MODEL_LAYERS["gin"] = _gin_layer
MODEL_NAMES = tuple(MODEL_LAYERS)
