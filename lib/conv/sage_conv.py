import torch
from torch import Tensor, Size
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from typing import Optional, Union
from torch_geometric.typing import PairTensor

from ..cache_system import LayerCache, LazyCache
from ..model import Basic
from ..profiler import Profiler

# clocks for profiler
MODEL_CLOCK = 101

class SAGEConvReduce(SAGEConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str = 'mean',
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            normalize=normalize,
            root_weight=root_weight,
            project=project,
            bias=bias,
            **kwargs
        )


    def preprocess(self, x: Tensor):
        if self.project and hasattr(self, 'lin'):
            return self.lin(x).relu()
        
        return x


    def part_forward(
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Optional[Tensor] = None,
            size: Optional[Size] = None,
    ):
        aggr_out = self.update(x[0])
        out = self.lin_l(aggr_out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class SAGE(Basic):
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
        return SAGEConvReduce(in_channels, out_channels, **kwargs)
    

    def preprocess(
            self,
            embd_cache: LazyCache,
            targets: Tensor,
            device: torch.device,
            p: Profiler,

            layer_idx: int,
    ):
        p.begin(MODEL_CLOCK)
        src_embd = embd_cache.fetch(
            targets=targets,
            device=device,
        )
        p.record('update.compute.part_computation.transmission', p.dur(MODEL_CLOCK))

        src_embd = self.convs[layer_idx].preprocess(src_embd)

        p.record('update.compute.part_computation.computation', p.dur(MODEL_CLOCK))

        return src_embd
    

    def part_compute(
            self,
            layer_idx: int,
            dst: Tensor,
            aggr_out: Tensor,
            embd_cache: LazyCache,
    ):
        x = embd_cache.fetch(
            targets=dst,
            device=aggr_out.device,
        )
        x = (aggr_out, x)
        y = self.convs[layer_idx].part_forward(x)
        if layer_idx != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y