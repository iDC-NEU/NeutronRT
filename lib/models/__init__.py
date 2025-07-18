import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

from dgl.heterograph import DGLBlock

from .convs import (
    GraphConvReduce,
    GINConvReduce,
    SAGEConvReduce,
    AGNNConvReduce,
    GMMConvReduce,
    GATConvReduce,
)

conv_map = {
    'gcn-partial': GraphConvReduce,
    'gin-partial': GINConvReduce,
    'sage-partial': SAGEConvReduce,
    'agnn-partial': AGNNConvReduce,
    'monet-partial': GMMConvReduce,
    'gat-partial': GATConvReduce,

    'gcn'  : dglnn.GraphConv,
    'gin'  : dglnn.GINConv,
    'sage' : dglnn.SAGEConv,
    'agnn' : dglnn.AGNNConv,
    'monet': dglnn.GMMConv,
    'gat'  : dglnn.GATConv,
}

class GNNModel(nn.Module):
    def __init__(self, model_name: str, num_features: int, num_hiddens: int, num_classes: int, num_layers: int, use_softmax: bool, **kwargs):
        super().__init__()

        self.model_name = model_name
        self.model_type = model_name.split('-')[0]
        self.use_softmax = use_softmax

        self.num_layers = num_layers
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        cls = conv_map.get(model_name.lower())
        for i in range(num_layers):
            in_hidden = num_features if i == 0 else num_hiddens
            out_hidden = num_classes if i == num_layers - 1 else num_hiddens
            self.layers.append(cls(in_hidden, out_hidden, **kwargs))


    def forward(self, layer_idx: int, block: DGLBlock, h: th.Tensor) -> th.Tensor:
        layer = self.layers[layer_idx]
        h_dst = h[:block.number_of_dst_nodes()]
        h = layer(block, (h, h_dst))

        a = None
        if isinstance(h, tuple):
            h, a = h
        
        if self.model_type == 'gat':
            h = h.mean(1)
        if layer_idx != self.num_layers - 1:
            h = F.relu(h)

        if a is not None:
            h = h, a

        return h