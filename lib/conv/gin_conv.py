import torch
from torch import Tensor, Size
from torch.nn import functional as F
from torch_geometric.nn import GINConv, MLP

from typing import Optional, Callable

from ..cache_system import LayerCache, LazyCache
from ..model import Basic

class GINConvReduce(GINConv):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            nn: Callable,
            eps: float = 0,
            train_eps: bool = False,
            **kwargs
    ):
        super().__init__(
            in_channel=in_channel,
            out_channel=out_channel,
            nn=nn,
            eps=eps,
            train_eps=train_eps,
            **kwargs,
        )


    def part_forward(
            self,
            x: Tensor,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        aggr_out = self.update(x[0])
        
        x_r = x[1]
        if x_r is not None:
            out = aggr_out + (1 + self.eps) * x_r

        return self.nn(out)


class GIN(Basic):
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
        kwargs.setdefault('nn', 'mlp')
        if kwargs.get('nn') == 'mlp':
            kwargs['nn'] = MLP(
                [in_channels, out_channels, out_channels],
                act=F.relu,
            )
        else:
            raise NotImplementedError

        return GINConvReduce(in_channels, out_channels, **kwargs)
    

    def part_compute(
            self,
            layer_idx: int,
            aggr_out: Tensor,
            **kwargs,
    ):
        y = self.convs[layer_idx].part_forward(aggr_out)
        if layer_idx != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y