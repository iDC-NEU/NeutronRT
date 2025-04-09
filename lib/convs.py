import logging
import torch
from torch import nn
from torch.nn import functional as F
import dgl.nn as dglnn
import dgl.function as fn
from dgl.convert import block_to_graph
from dgl.transforms import reverse
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.nn.functional import edge_softmax

# types
from dgl import DGLGraph
from dgl.heterograph import DGLBlock
from dgl.base import DGLError
from torch import Tensor
from typing import Callable, Optional

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
                graph.dstdata["neigh"] = graph.dstdata["h"]
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                graph.dstdata["h"].add_(graph.dstdata["aggr_cache"])
                graph.dstdata["neigh"] = graph.dstdata["h"]
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = torch.matmul(rst, weight)

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
        

class GINConvReduce(dglnn.GINConv):
    def __init__(
            self,
            apply_func: Optional[Callable] = None,
            aggregator_type: str = "sum",
            init_eps: float = 0,
            learn_eps: bool = False,
            activation: Optional[Callable] = None,
    ):
        super().__init__(apply_func, aggregator_type, init_eps, learn_eps, activation)


    def forward(self, graph: DGLGraph, feat: Tensor, edge_weight = None):
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, _reducer("m", "neigh"))
            
            graph.dstdata["neigh"].add_(graph.dstdata["aggr_cache"])
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]

            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst


class SAGEConvReduce(dglnn.SAGEConv):
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            aggregator_type: str,
            feat_drop: float = 0.0,
            bias: bool = True,
            norm: Optional[Callable] = None,
            activation: Optional[Callable] = None,
    ):
        super().__init__(in_feats, out_feats, aggregator_type, feat_drop, bias, norm, activation)


    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                # graph.update_all(msg_fn, fn.mean("m", "neigh"))
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                expend_sum = graph.dstdata["aggr_cache"] * graph.dstdata["old_deg"]
                graph.dstdata["neigh"] = (graph.dstdata["neigh"] + expend_sum) / graph.dstdata["new_deg"]
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                graph.dstdata["neigh"].add_(graph.dstdata["aggr_cache"])
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                graph.dstdata["neigh"] = torch.max(graph.dstdata["neigh"], graph.dstdata["aggr_cache"])
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                raise NotImplementedError
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
        

# TODO: implement
class AGNNConvReduce(dglnn.AGNNConv):
    def __init__(
        self,
        init_beta: float = 1.0,
        learn_beta: bool = True,
        allow_zero_in_degree: bool = False,
    ):
        super().__init__(init_beta, learn_beta, allow_zero_in_degree)

    
    def forward(self, graph, feat):
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

            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata["h"] = feat_src
            graph.srcdata["norm_h"] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata["norm_h"] = F.normalize(feat_dst, p=2, dim=-1)
            # compute cosine distance
            graph.apply_edges(fn.u_dot_v("norm_h", "norm_h", "cos"))
            cos = graph.edata.pop("cos")
            e = self.beta * cos
            graph.edata["p"] = edge_softmax(graph, e)
            graph.update_all(fn.u_mul_e("h", "p", "m"), fn.sum("m", "h"))
            return graph.dstdata.pop("h")
        

# TODO: implement
class GMMConvReduce(dglnn.GMMConv):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        dim: int,
        n_kernels: int,
        aggregator_type: str = "sum",
        residual: bool = False,
        bias: bool = True,
        allow_zero_in_degree: bool = False,
    ):
        super().__init__(in_feats, out_feats, dim, n_kernels, aggregator_type, residual, bias, allow_zero_in_degree)


    def forward(self, graph, feat, pseudo):
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

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = self.fc(feat_src).view(
                -1, self._n_kernels, self._out_feats
            )
            E = graph.num_edges()
            # compute gaussian weight
            gaussian = -0.5 * (
                (
                    pseudo.view(E, 1, self._dim)
                    - self.mu.view(1, self._n_kernels, self._dim)
                )
                ** 2
            )
            gaussian = gaussian * (
                self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2
            )
            gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # (E, K, 1)
            graph.edata["w"] = gaussian
            graph.update_all(fn.u_mul_e("h", "w", "m"), self._reducer("m", "h"))
            rst = graph.dstdata["h"].sum(1)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


# TODO: implement
class GATConvReduce(dglnn.GATConv):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super().__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias)


    def forward(self, graph, feat, edge_weight=None, get_attention=False):
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

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
