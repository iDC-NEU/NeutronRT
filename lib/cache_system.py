from .cli import CommandLineArgument
from .graph_dataset import GraphDataset
from .profiler import Profiler
from .utils import free_ram

import torch, logging
from torch_scatter import scatter

from typing import Optional, List

external_logger = logging.getLogger('cache hit')
CACHE_CLK = 201

class LayerCache():
    def __init__(
            self,
            num_dim: int,
            layer_idx: int,
            cache_type: str,
            cuda_cache: torch.Tensor,
            cuda_key: Optional[torch.Tensor],
            host_cache: Optional[torch.Tensor],
            spec: str = 'none',
    ):
        self.num_dim = num_dim
        self.layer_idx = layer_idx
        self.cache_type = cache_type
        self.hot_stream = torch.cuda.Stream()

        if host_cache == None and cuda_key == None:
            self.in_memory = True
            device = cuda_cache.device
            length = cuda_cache.size(0)

            self.cuda: torch.Tensor = cuda_cache
            self.dtype = cuda_cache.dtype
        else:
            self.in_memory = False
            device = host_cache.device
            length = host_cache.size(0)

            self.cuda: torch.Tensor = cuda_cache
            self.cuda_key: torch.Tensor = cuda_key
            self.host: torch.Tensor = host_cache
            self.hotness: torch.Tensor = torch.zeros(length, dtype=torch.float)
            self.hot_indice: torch.Tensor = torch.arange(length, dtype=torch.int64)
            self.dtype = host_cache.dtype

        self.spec: str = spec


    def fetch(
            self,
            targets: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        # self.hotness.index_add_(dim=0, index=targets.cpu(), source=torch.ones_like(targets.cpu(), dtype=torch.float))

        if self.in_memory:
            if not targets.is_cuda:
                targets = targets.to(self.cuda.device)
            return self.cuda[targets].to(**kwargs)
        
        if self.cuda is not None and self.cuda.size(0) > 0:
            ret_cache = torch.zeros(
                (targets.size(0), self.host.size(1)),
                dtype=self.host.dtype,
            ).to(**kwargs)

            # first fetch from cuda cache
            key = torch.searchsorted(self.cuda_key, targets, side='right').sub(1)
            mask = self.cuda_key[key] == targets
            ret_cache[mask] = self.cuda[key[mask]]

            mask = ~mask
            targets = targets[mask]
            if not targets.is_cpu:
                targets = targets.cpu()
            ret_cache[mask] = self.host[targets].to(**kwargs)
        else:
            if not targets.is_cpu:
                targets = targets.cpu()
            ret_cache = self.host[targets].to(**kwargs)

        return ret_cache
        # return self.host[targets].to(**kwargs)


    def update(
            self,
            targets: torch.Tensor,
            new_cache: torch.Tensor,
            non_blocking: bool = False,
    ):
        # ensure that targets is the index of new_cache
        # assuming that targets is unique and sorted

        if self.in_memory:
            if not targets.is_cuda:
                targets = targets.to(self.cuda.device)
            return self.cuda[targets].copy_(new_cache, non_blocking)

        if not targets.is_cpu:
            self.host[targets.cpu()].copy_(new_cache, non_blocking)
        else:
            self.host[targets].copy_(new_cache, non_blocking)

        if self.cuda is not None and self.cuda.size(0) > 0:
            key = torch.searchsorted(self.cuda_key, targets, side='right').sub(1)
            mask = self.cuda_key[key] == targets
            self.cuda[key[mask]].copy_(new_cache[mask])


    # def clone(self):
    #     if self.in_memory:
    #         return LayerCache(
    #             cuda_cache=self.cuda.clone(),
    #             cuda_key=None,
    #             host_cache=None,
    #             spec=self.spec,
    #         )
    #     return LayerCache(
    #         cuda_cache=self.cuda.clone(),
    #         cuda_key=self.cuda_key.clone(),
    #         host_cache=self.host.clone(),
    #         spec=self.spec,
    #     )
    

    def replace(self):
        self.hotness, indice = self.hotness.sort(descending=True)
        self.hot_indice = self.hot_indice[indice]
        topk, _ = self.hot_indice[:self.cuda_key.size(0)].sort()
        key = torch.searchsorted(self.cuda_key, topk, side='right').sub(1)
        mask = self.cuda_key[key] == topk

        if not mask.all():
            self.cuda[key[~mask]].copy_(self.host[topk[~mask]])
            self.cuda_key[key[~mask]] = topk[~mask]

            self.cuda_key, indice = self.cuda_key.sort()
            self.cuda = self.cuda[indice]


class LazyCache():
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.targets: torch.Tensor = None
        self.cache: torch.Tensor = None
        self.upstream: LayerCache = None
        self.fetch_old: bool = False


    def set_upstream(self, lc: LayerCache):
        self.upstream = lc
    

    def update(self, targets: torch.Tensor, cache: torch.Tensor):
        if self.targets is None:
            targets, indice = targets.sort()
            assert self.cache is None, 'LazyCache failed'
            if targets.device != self.device:
                targets = targets.to(self.device)
                cache = cache[indice].to(self.device)
            self.targets = targets
            self.cache = cache
        else:
            if targets.device != self.device:
                targets = targets.to(self.device)
                cache = cache.to(self.device)
            self.targets, indice = torch.cat([self.targets, targets]).sort()
            self.cache = torch.cat([self.cache, cache])[indice]


    def do_update(self):
        if self.targets is not None:
            self.upstream.update(
                targets=self.targets,
                new_cache=self.cache,
                non_blocking=True,
            )

        self.targets: torch.Tensor = None
        self.cache: torch.Tensor = None


    def fetch(
            self,
            targets: torch.Tensor,
            p: Profiler,
            **kwargs,
    ):
        targets, indice = targets.unique(return_inverse=True)
        p.record('load_embd', targets.size(0))

        if self.fetch_old:
            p.begin(CACHE_CLK)
            cache = self.old_fetch(targets, **kwargs)
            p.record('old fetch', p.dur(CACHE_CLK))
        else:
            p.begin(CACHE_CLK)
            cache = self.new_fetch(targets, **kwargs)
            p.record('new fetch', p.dur(CACHE_CLK))
        return cache[indice]


    def new_fetch(
            self,
            targets: torch.Tensor,
            **kwargs,
    ):
        if self.targets is None:
            return self.old_fetch(targets, **kwargs)

        if targets.device != self.device:
            targets = targets.to(self.device)

        index = torch.arange(
            targets.size(0),
            dtype=torch.int64,
            device=self.device,
        )

        # first fetch from cuda cache
        key = torch.searchsorted(self.targets, targets, side='right').sub(1)
        mask = self.targets[key] == targets

        lazy = self.cache[key[mask]].to(**kwargs)
        old = self.old_fetch(targets[~mask], **kwargs)

        index = torch.cat([index[mask], index[~mask]]).to(**kwargs)
        cache = torch.cat([lazy, old])
        cache = scatter(cache, index, dim=0, dim_size=targets.size(0))

        # self.upstream.hotness.index_add_(dim=0, index=targets[mask].cpu(), source=torch.ones_like(targets[mask], dtype=torch.float).cpu())

        return cache.contiguous()
    

    def old_fetch(
            self,
            targets: torch.Tensor,
            **kwargs,
    ):
        return self.upstream.fetch(
            targets=targets,
            **kwargs,
        )


class CacheSystem():
    def __init__(self, config: CommandLineArgument, gds: GraphDataset):
        self.num_nodes = gds.data.num_nodes
        self.num_model_layers = config.num_model_layers
        self.device = config.device
        self.aggr_method = config.aggr_method

        self.embd_host_ratio = [r if r <= 1.0 else min(r / self.num_nodes, 1.0) for r in config.embd_host_ratio]
        self.embd_cuda_ratio = [r if r <= 1.0 else min(r / self.num_nodes, 1.0) for r in config.embd_cuda_ratio]
        self.aggr_host_ratio = [r if r <= 1.0 else min(r / self.num_nodes, 1.0) for r in config.aggr_host_ratio]
        self.aggr_cuda_ratio = [r if r <= 1.0 else min(r / self.num_nodes, 1.0) for r in config.aggr_cuda_ratio]

        self.embd_cache: List[LayerCache] = []
        self.aggr_cache: List[LayerCache] = []

        self.logger = logging.getLogger('cache')
        self.logger.debug(f'{free_ram():.2f}GB already allocated')

        if self.aggr_method == 'mean':
            self.in_degree: torch.Tensor = gds.pma.in_degrees().clone().to(config.device)
            self.out_degree: torch.Tensor = gds.pma.out_degrees().clone().to(config.device)

        _, sorted_in_degree = self.in_degree.sort(descending=True)
        _, sorted_out_degree = self.out_degree.sort(descending=True)

        # assuming host can always hold all feat
        if self.embd_cuda_ratio[0] < 1:
            self.embd_in_memory = False
            
            # create host embedding cache
            host_embd = gds.data.x.pin_memory()

            # create cuda embedding cache
            num_embd_cuda = int(self.embd_cuda_ratio[0] * self.num_nodes)
            embd_cuda_key, _ = sorted_out_degree[:num_embd_cuda].cpu().sort()
            cuda_embd = gds.data.x[embd_cuda_key].to(config.device, non_blocking=True)
            embd_cuda_key = embd_cuda_key.to(config.device, non_blocking=True)
        else:
            self.embd_in_memory = True
            host_embd = None
            embd_cuda_key = None
            
            cuda_embd = gds.data.x.to(config.device, non_blocking=True)

        self.embd_cache.append(
            LayerCache(
                num_dim=gds.data.x.size(1),
                layer_idx=0,
                cache_type='embd',
                cuda_cache=cuda_embd,
                cuda_key=embd_cuda_key,
                host_cache=host_embd,
            )
        )

        for layer_idx in range(config.num_model_layers):
            num_dim =  config.num_hiddens if layer_idx != config.num_model_layers - 1 else gds.data.num_classes

            if config.num_model_layers - layer_idx > 1:
                if self.embd_cuda_ratio[layer_idx] < 1:
                    self.embd_in_memory = False
                
                    # create host embedding cache
                    host_embd = torch.zeros((self.num_nodes, num_dim), dtype=gds.data.x.dtype, pin_memory=True)

                    # create cuda embedding cache
                    num_embd_cuda = int(self.embd_cuda_ratio[layer_idx] * self.num_nodes)
                    embd_cuda_key, _ = sorted_out_degree[:num_embd_cuda].cpu().sort()
                    
                    cuda_embd = torch.zeros((num_embd_cuda, num_dim), dtype=gds.data.x.dtype, device=config.device)
                    
                    embd_cuda_key = embd_cuda_key.to(config.device, non_blocking=True)
                else:
                    self.embd_in_memory = True
                    host_embd = None
                    embd_cuda_key = None

                    # create cuda embedding cache
                    cuda_embd = torch.zeros((self.num_nodes, num_dim), dtype=gds.data.x.dtype, device=config.device)

                self.embd_cache.append(
                    LayerCache(
                        num_dim=num_dim,
                        layer_idx=layer_idx + 1,
                        cache_type='embd',
                        cuda_cache=cuda_embd,
                        cuda_key=embd_cuda_key,
                        host_cache=host_embd,
                    )
                )

            if self.aggr_cuda_ratio[layer_idx] < 1:
                self.aggr_in_memory = False

                # create host embedding cache
                host_aggr = torch.zeros((self.num_nodes, num_dim), dtype=gds.data.x.dtype, pin_memory=True)

                # create cuda embedding cache
                num_aggr_cuda = int(self.aggr_cuda_ratio[layer_idx] * self.num_nodes)
                aggr_cuda_key, _ = sorted_in_degree[:num_aggr_cuda].to(config.device, non_blocking=True).sort()

                cuda_aggr = torch.zeros((num_aggr_cuda, num_dim), dtype=gds.data.x.dtype, device=config.device)
            else:
                self.aggr_in_memory = True
                host_aggr = None
                aggr_cuda_key = None

                # create cuda embedding cache
                cuda_aggr = torch.zeros((self.num_nodes, num_dim), dtype=gds.data.x.dtype, device=config.device)


            if config.model == 'gat':
                spec = 'softmax'
            else:
                spec = 'none'

            self.aggr_cache.append(
                LayerCache(
                    num_dim=num_dim,
                    layer_idx=layer_idx,
                    cache_type='aggr',
                    cuda_cache=cuda_aggr,
                    cuda_key=aggr_cuda_key,
                    host_cache=host_aggr,
                    spec=spec,
                )
            )

            self.logger.debug(f'{free_ram():.2f}GB already allocated')


    def in_degrees(self):
        return self.in_degree


    def update_in_degree(self, gds: GraphDataset):
        self.in_degree.copy_(gds.pma.clone_degree())
                
                

