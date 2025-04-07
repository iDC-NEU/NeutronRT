import logging
import torch

# types
from dgl import DGLGraph
from typing import List

# self defined
from lib.config import Config
from lib.utils import tsstr


class LayerCache():
    def __init__(
            self,
            cache: torch.Tensor
    ):
        logging.info(f'create cache: {tsstr(cache)}')
        self.cache = cache

    def fetch(self, targets: torch.Tensor, device: torch.device):
        if targets.device != self.cache.device:
            targets = targets.to(self.cache.device)
        targets, indice = targets.unique(return_inverse=True)
        cache = self.cache[targets]
        if device != self.cache.device:
            cache = cache.to(device)
        cache = cache[indice]
        return cache


class Cache():
    def __init__(
            self,
            cfg: Config,
            g: DGLGraph,
    ):
        self.device = 'cuda' if cfg.in_memory else 'cpu'
        self.h_cache: List[LayerCache] = []
        self.a_cache: List[LayerCache] = []
        
        logging.info(f'cache device: {self.device}')

        self.h_cache.append(
            LayerCache(g.ndata['feat'].to(self.device))
        )

        for i in range(cfg.num_layers - 1):
            self.h_cache.append(
                LayerCache(
                    torch.rand(
                        (g.num_nodes(), cfg.num_hiddens),
                        dtype=g.ndata['feat'].dtype,
                        device=self.device,
                        pin_memory=not cfg.in_memory,
                    )
                )
            )


    def build_aggr_cache(
            self,
            cfg: Config,
            g: DGLGraph,
    ):
        for i in range(cfg.num_layers):
            num_dim = cfg.num_hiddens if cfg.num_layers - i > 1 else cfg.num_classes

            self.a_cache.append(
                LayerCache(
                    torch.rand(
                        (g.num_nodes(), num_dim),
                        dtype=g.ndata['feat'].dtype,
                        device=self.device,
                        pin_memory=not cfg.in_memory,
                    )
                )
            )
