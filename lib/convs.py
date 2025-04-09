import logging
import torch
from torch import nn
import dgl.nn as dglnn
import dgl.function as fn
from dgl.convert import block_to_graph
from dgl.transforms import reverse
from dgl.utils import expand_as_pair

# types
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from dgl.base import DGLError
from torch import Tensor
from typing import Callable

# self defined
from .cache import Cache, LayerCache
from .utils import tsstr


class GraphConvReduce(dglnn.GraphConv):
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            norm: str = 'both',
            weight: bool = True,
            bias: bool = True,
            activation: Callable = None,
            allow_zero_in_degree: bool = False,
    ):
        super().__init__(in_feats, out_feats, norm, weight, bias, activation, allow_zero_in_degree)
        

    def forward(self, graph: DGLGraph, feat: Tensor, weight = None, edge_weight = None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                graph.dstdata["h"].add_(graph.dstdata["aggr_cache"])
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = torch.matmul(rst, weight)
                rst.add_(graph.dstdata["aggr_cache"])

            # save aggregation output
            graph.dstdata["aggr_cache"] = rst

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
        

class GATv2ConvReduce():
    def __init__(self) -> None:
        pass
