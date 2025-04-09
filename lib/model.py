import torch, logging

from torch import nn, Tensor
from torch.nn import functional as F
from torch_scatter import scatter

from typing import Optional, Tuple
from torch_geometric.typing import PairTensor

from .cache_system import CacheSystem, LayerCache, LazyCache
from .profiler import Profiler
from .graph_dataset import GraphDataset


# clocks for profiler
MODEL_CLOCK = 101

aggr_map = {
    'amax': 'max',
    'amin': 'min',
}


def dst_based_sort(edge_index: Tensor, uptype: Tensor):
    src, dst = edge_index
    dst, indice, counts = dst.unique(
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    indice, reorder = indice.sort()
    src = src[reorder]
    uptype = uptype[reorder]
    return src, dst, indice, counts, uptype


def compute_batch_range(batch_size: int, nums: Tensor):
    begin = 0
    cum_sum = torch.cumsum(nums, dim=0)
    length = len(cum_sum)
    while begin < length:
        target = cum_sum[begin] + batch_size
        end = torch.searchsorted(cum_sum[begin:], target, side='right')
        end = end.item() + begin
        r = cum_sum[begin:end]
        cbegin, cend = r[0], r[-1]
        end = end - 1
        if end == begin:
            break
        yield (begin, end, cbegin, cend)
        begin = end


class Basic(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int,
            device: torch.device,
            bs_compute: int,
            **kwargs,
    ):
        super(Basic, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.device = device
        self.bs_compute = bs_compute
        self.logger = logging.getLogger('model')

        if kwargs.get('aggr', None) is None:
            kwargs.setdefault('aggr', 'mean')
        aggr_method = kwargs.get('aggr')
        self.aggr_method = aggr_map.get(aggr_method, aggr_method)

        self.convs = nn.ModuleList()
        self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_channels, hidden_channels, **kwargs))
        self.convs.append(self.init_conv(hidden_channels, out_channels, **kwargs))

        self.to(device)


    def init_conv(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs,
    ):
        raise NotImplementedError
    

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def part_batch_compute(
            self,
            layer_idx: int,
            update_subgraph: Tensor,
            update_type: Tensor,
            now_embd_cache: Optional[LayerCache],
            gds: GraphDataset,
            csys: CacheSystem,
            p: Profiler,
    ) -> Tensor:
        if now_embd_cache is None:
            now_embd_cache = LazyCache(gds.data.x.device)
            now_embd_cache.set_upstream(csys.embd_cache[layer_idx])

        # embd_cache = csys.embd_cache[layer_idx]
        # next_embd_cache = csys.embd_cache[layer_idx + 1]
        # old_embd_cache = csys.snapshot_embedding(layer_idx + 1)
        aggr_cache = csys.aggr_cache[layer_idx]

        src, dst, indice, counts, uptype = dst_based_sort(update_subgraph, update_type)

        spec_dict = {}
        if self.aggr_method == 'mean':
            spec_dict['old_in_degree'] = csys.in_degrees()
            spec_dict['new_in_degree'] = gds.pma.in_degrees()

        nxt_embd_cache = LazyCache(gds.data.x.device)
        if self.num_layers - layer_idx > 1:
            nxt_embd_cache.set_upstream(csys.embd_cache[layer_idx + 1])

        for begin, end, cbegin, cend in compute_batch_range(self.bs_compute, counts):
            p.record('nbatches', 1)

            batch_dst = dst[begin:end]
            batch_src = src[cbegin:cend]
            batch_indice = indice[cbegin:cend].sub(indice[cbegin])
            batch_uptype = uptype[cbegin:cend]

            model_spec_dict = {}
            if self.__class__.__name__ == 'SAGE':
                model_spec_dict['layer_idx'] = layer_idx
            elif self.__class__.__name__ == 'GCN':
                model_spec_dict['layer_idx'] = layer_idx
                model_spec_dict['dst'] = dst
            
            if layer_idx == 0:
                p.begin(MODEL_CLOCK)
                if self.__class__.__name__ == 'GCN':
                    model_spec_dict['indice'] = batch_indice

                now_embd_cache.fetch_old = True
                src_embd = self.preprocess(
                    embd_cache=now_embd_cache,
                    targets=batch_src,
                    device=self.device,
                    p=p,
                    **model_spec_dict,
                )
            else:
                p.begin(MODEL_CLOCK)
                use_old_embd = batch_uptype <= 0
                src_use_old = batch_src[use_old_embd]

                if self.__class__.__name__ == 'GCN':
                    model_spec_dict['indice'] = batch_indice[use_old_embd]

                now_embd_cache.fetch_old = True
                src_embd_old = self.preprocess(
                    embd_cache=now_embd_cache,
                    targets=src_use_old,
                    device=self.device,
                    p=p,
                    **model_spec_dict,
                )

                use_new_embd = batch_uptype >= 0
                src_use_new = batch_src[use_new_embd]

                if self.__class__.__name__ == 'GCN':
                    model_spec_dict['indice'] = batch_indice[use_new_embd]

                now_embd_cache.fetch_old = False
                src_embd_new = self.preprocess(
                    embd_cache=now_embd_cache,
                    targets=src_use_new,
                    device=self.device,
                    p=p,
                    **model_spec_dict,
                )

                batch_indice = torch.cat([batch_indice[use_old_embd], batch_indice[use_new_embd]])
                src_embd = (src_embd_old, src_embd_new)

            batch_aggr = self.batch_aggregation(
                dst=batch_dst,
                src_embd=src_embd,
                indice=batch_indice,
                uptype=batch_uptype,
                aggr_cache=aggr_cache,
                p=p,
                **spec_dict,
            )

            p.begin(MODEL_CLOCK)
            now_embd_cache.fetch_old = False
            batch_embd: Tensor = self.part_compute(
                layer_idx=layer_idx,
                dst=batch_dst,
                aggr_out=batch_aggr,
                embd_cache=now_embd_cache,
            )
            p.record('computation', p.dur(MODEL_CLOCK))

            del batch_aggr

            nxt_embd_cache.update(
                targets=batch_dst,
                cache=batch_embd,
            )
            del batch_embd
            p.record('transmission', p.dur(MODEL_CLOCK))

        now_embd_cache.do_update()
        p.record('transmission', p.dur(MODEL_CLOCK))
        
        return dst, nxt_embd_cache


    def batch_aggregation(
            self,
            dst: Tensor,
            src_embd: PairTensor,
            indice: Tensor,
            uptype: Tensor,
            aggr_cache: LayerCache,
            p: Profiler,
            old_in_degree: Optional[Tensor] = None,
            new_in_degree: Optional[Tensor] = None,
    ) -> Tensor:
        p.begin(MODEL_CLOCK)
        # for deleted and modified edges
        if isinstance(src_embd, Tensor):
            src_embd.mul_(uptype.view(-1, 1))
        elif isinstance(src_embd, Tuple):
            src_embd[0].neg_()
            src_embd = torch.cat(src_embd)
        else:
            raise NotImplementedError
        
        aggr_out: Tensor = scatter(
            src=src_embd,
            index=indice,
            dim=0,
            reduce=self.aggr_method,
        )
        p.record('computation', p.dur(MODEL_CLOCK))
        del src_embd

        p.begin(MODEL_CLOCK)
        old_aggr = aggr_cache.fetch(
            targets=dst,
            device=self.device,
        )
        p.record('transmission', p.dur(MODEL_CLOCK))
        p.record('load_aggr', dst.size(0))

        if self.aggr_method == 'mean':
            old_aggr.mul_(old_in_degree[dst].view(-1, 1))
            aggr_out.add_(old_aggr).div_(new_in_degree[dst].view(-1, 1))
        elif self.aggr_method == 'sum':
            aggr_out.add_(old_aggr)
        else:
            raise NotImplementedError

        p.record('computation', p.dur(MODEL_CLOCK))

        del old_aggr
        p.begin(MODEL_CLOCK)
        aggr_cache.update(
            targets=dst,
            new_cache=aggr_out,
            non_blocking=True,
        )
        p.record('transmission', p.dur(MODEL_CLOCK))
        p.record('store_aggr', dst.size(0))

        return aggr_out
    

    def preprocess(
            self,
            embd_cache: LazyCache,
            targets: Tensor,
            device: torch.device,
            p: Profiler,
            **kwargs,
    ):
        """
        Preprocessing before aggregation on source embeddings for some model
        """
        p.begin(MODEL_CLOCK)
        src_embd = embd_cache.fetch(
            targets=targets,
            device=device,
            p=p,
        )
        p.record('transmission', p.dur(MODEL_CLOCK))
        return src_embd
        

    def full_compute(
            self,
            layer_idx: int,
            x: Tensor,
            edge_index: Tensor,
    ):
        y = self.convs[layer_idx](
            x=x,
            edge_index=edge_index,
        )
        if layer_idx != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y
    

    def part_compute(
            self,
            layer_idx: int,
            dst: Tensor,
            aggr_out: Tensor,
            embd_cache: LazyCache,

    ):
        raise NotImplementedError


    