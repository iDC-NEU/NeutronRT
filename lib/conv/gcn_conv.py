import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter

from typing import Optional, Union
from torch_geometric.typing import PairTensor, SparseTensor

from ..cache_system import LayerCache, LazyCache
from ..model import Basic
from ..profiler import Profiler

# clocks for profiler
MODEL_CLOCK = 101

class GCNConvReduce(GCNConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: Optional[bool] = None,
            normalize: bool = True,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            improved=improved, 
            cached=cached, 
            add_self_loops=add_self_loops, 
            normalize=normalize, 
            bias=bias, 
            **kwargs
        )

        self.edge_weight = None


    def preprocess(
            self, 
            x: Tensor,
            edge_index: Tensor,
            edge_weight: Optional[Tensor] = None,
    ):
        if self.normalize:
            num_nodes = maybe_num_nodes(edge_index)

            if self.edge_weight is None:
                self.edge_weight = torch.ones(
                    edge_index.size(1),
                    dtype=x.dtype,
                    device=edge_index.device
                )
            edge_weight = self.edge_weight

            src, dst = edge_index
            deg = scatter(edge_weight, dst, dim=0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[src] * edge_weight * deg_inv_sqrt[dst]

        x = self.lin(x)

        return edge_weight.view(-1, 1) * x
        

    def part_forward(
            self,
            x: Union[Tensor, PairTensor],
    ):

        out = self.update(x)

        if self.bias is not None:
            out = out + self.bias

        return out


class GCN(Basic):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int,
            **kwargs,
        ):
        super().__init__(
            in_channels, 
            hidden_channels,
            out_channels,
            num_layers,
            **kwargs,
        )


    def init_conv(self, in_channels, out_channels, **kwargs):
        return GCNConvReduce(in_channels, out_channels, **kwargs)
    

    def preprocess(
            self,
            embd_cache: LazyCache,
            targets: Tensor,
            device: torch.device,
            p: Profiler,

            layer_idx: int,
            dst: Tensor,
            indice: Tensor,
    ):
        subgraph = torch.cat([targets, dst[indice]])
        _, subgraph = subgraph.unique(return_inverse=True)
        subgraph = subgraph.reshape(2, -1)

        p.begin(MODEL_CLOCK)
        x = embd_cache.fetch(
            targets=subgraph[0],
            device=device,
            p=p,
        )
        p.record('transmission', p.dur(MODEL_CLOCK))

        src_embd = self.convs[layer_idx].preprocess(
            x=x,
            edge_index=subgraph,
        )
        p.record('computation', p.dur(MODEL_CLOCK))
        
        return src_embd
    

    def part_compute(
            self,
            layer_idx: int,
            aggr_out: Tensor,
            **kwargs,
    ):
        x = aggr_out
        y = self.convs[layer_idx].part_forward(
            x=x,
        )
        if layer_idx != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y
