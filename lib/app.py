import timeit
import logging

import dgl
import torch
import torch.nn as nn
import torch.distributed as dist


# types
from typing import List
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler
from torch.nn.parallel import DistributedDataParallel

# self defined
from .cache import Cache
from .config import Config
from .utils import (
    free_ram,
    tsstr,
    PERF_LOG,
)

def r_sample(pidx: int, cfg: Config, g: DGLGraph, model: nn.Module, full_neighbour: bool = False, use_cache: bool = False):
    # find corresponding device for this rank
    device = cfg.gpus[pidx]
    torch.cuda.set_device(device)

    if cfg.in_memory:
        g = g.to('cuda')

    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12340",
        world_size=len(cfg.gpus),
        rank=pidx,
    )

    if full_neighbour:
        fanouts = [-1] * cfg.num_layers
    else:
        fanouts = cfg.fanouts

    model = model.to('cuda')
    model = DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )

    infer_test = f'SMP {fanouts}{" CACHE " if use_cache else " "}inference'

    if use_cache:
        caches = Cache(cfg, g)
        
        samplers: List[NeighborSampler] = []
        for layer_idx in range(cfg.num_layers):
            samplers.append(NeighborSampler(
                [cfg.fanouts[layer_idx]],
            ))

        logging.log(PERF_LOG, f'> {infer_test} begin')
        t = timeit.timeit(lambda: model.module.sample_cached_infer(cfg, g, samplers, caches), number=1)
        logging.log(PERF_LOG, f'{infer_test} in total: {t}')
        logging.log(PERF_LOG, f'< {infer_test} end')
    else:
        sampler = NeighborSampler(
            fanouts,
        )

        logging.log(PERF_LOG, f'> {infer_test} begin')
        t = timeit.timeit(lambda: model.module.sample_infer(cfg, g, sampler), number=1)
        logging.log(PERF_LOG, f'{infer_test} in total: {t}')
        logging.log(PERF_LOG, f'< {infer_test} end')

    logging.log(PERF_LOG, f'-----')
    dist.destroy_process_group()


def r_partial(pidx: int, cfg: Config, g: DGLGraph, model: nn.Module):
    infer_test = 'INC inference'

    # find corresponding device for this rank
    device = cfg.gpus[pidx]
    torch.cuda.set_device(device)

    if cfg.in_memory:
        g = g.to('cuda')

    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12340",
        world_size=len(cfg.gpus),
        rank=pidx,
    )

    logging.info('build cache')
    fram = free_ram()
    caches = Cache(cfg, g)
    caches.build_aggr_cache(cfg, g)
    logging.info(f'cache built with {free_ram() - fram} GB memory used')

    model = model.to('cuda')
    model = DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )

    logging.log(PERF_LOG, f'> {infer_test}')
    t = timeit.timeit(lambda: model.module.partial_infer(cfg, g, caches), number=1)
    logging.log(PERF_LOG, f'{infer_test} in total: {t}')
    logging.log(PERF_LOG, f'< {infer_test}')

    logging.log(PERF_LOG, f'-----')
    dist.destroy_process_group()