import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter

from typing import Optional, Tuple
from torch_geometric.typing import SparseTensor, torch_sparse, PairTensor

from ..cache_system import LazyCache
from ..model import Basic


class GATv2ConvReduce(GATv2Conv):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            heads = 1, 
            concat = False, 
            negative_slope = 0.2, 
            dropout = 0, 
            add_self_loops = True,
            edge_dim = None, 
            fill_value = 'mean', 
            bias = True, 
            share_weights = False, 
            residual = False, 
            **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            share_weights=share_weights,
            residual=residual,
            **kwargs
        )

    
    def preprocess(self, x: Tensor):
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: Optional[Tensor] = None
        x_r: Optional[Tensor] = None

        assert x.dim() == 2

        if self.res is not None:
            res = self.res(x)

        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # not add_self_loops in this implementation

        x = x_l + x_r
        alpha = (x * self.att).sum(dim=-1)

        # TODO: support for concat
        x = x.mean(1)
        alpha = alpha.exp()
        alpha = alpha.mean(1)
        
        return x, alpha, res


    def part_forward( 
            self,
            x: Tensor,
            res: Tensor,
    ):
        out = self.update(x)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        return out


class GAT(Basic):
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
        return GATv2ConvReduce(in_channels, out_channels, **kwargs)
    

    def preprocess(
            self,
            layer_idx: int,
            embd_cache: LazyCache,
            targets: Tensor,
            device: torch.device
    ):
        src_embd = embd_cache.fetch(
            targets=targets,
            device=device,
        )

        return self.convs[layer_idx].preprocess(src_embd)


    def batch_aggregation(
            self,
            dst: Tensor,
            src_embd: Tuple[Tuple[Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Tensor, Optional[Tensor]]],
            indice: Tensor,
            uptype: Tensor,
            aggr_cache: LazyCache,
            old_in_degree: Optional[Tensor] = None,
            new_in_degree: Optional[Tensor] = None,
    ) -> Tensor:
        assert isinstance(src_embd, Tuple)

        if len(src_embd) == 2:
            delete_embd, insert_embd = src_embd
            delete_x, delete_alpha, delete_res = delete_embd
            insert_x, insert_alpha, insert_res = insert_embd
            x = torch.cat([delete_x.neg_(), insert_x])
            alpha = torch.cat([delete_alpha, insert_alpha])
            uptype = torch.ones(x.size(0), device=alpha.device)
            uptype[:delete_x.size(0)].neg_()

            if delete_res is not None and insert_res is not None:
                res = torch.cat([delete_res, insert_res])
            else:
                res = None
        elif len(src_embd) == 3:
            x, alpha, res = src_embd
            x = x * uptype.view(-1, 1)

        new_lower = scatter(uptype * alpha, indice, 0, reduce='sum')
        new_upper = scatter(x * alpha.view(-1, 1), indice, 0, reduce='sum')

        old_esum = aggr_cache.spec_fetch(
            targets=dst,
            device=self.device,
        )

        old_upper, old_lower = old_esum

        new_upper = old_upper + new_upper
        new_lower = old_lower + new_lower.view(-1, 1)
        new_cache = torch.cat([new_upper.unsqueeze(0), new_lower.unsqueeze(0)])

        aggr_cache.spec_update(
            targets=dst,
            new_cache=new_cache,
            non_blocking=True,
        )

        aggr_out = new_upper / new_lower

        return aggr_out, res


    def part_compute(
            self,
            layer_idx: int,
            aggr_out: Tensor,
            **kwargs,
    ):
        x, res = aggr_out
        y = self.convs[layer_idx].part_forward(x, res)
        if layer_idx != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y