import logging, os, random, sys, torch, numpy

# system initialization arguments
random.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)

# set environment variable
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['NUMEXPR_MAX_THREADS'] = '40'

# set logger file
logging.basicConfig(
    filename='logs/compute-4-part.log',
    filemode='w',
    level=logging.DEBUG,
)

import torch, resource
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from typing import Optional, List, Generator, Tuple

from lib.graph_dataset import GraphDataset
from lib.cli import CommandLineArgument
from lib.profiler import Profiler
from lib.cache_system import CacheSystem, LazyCache
from lib.task import Task

# clocks for profiler
DEFAULT_CLK = 1
COMPUTE_CLK = 2

# global __main__ logger
logger = logging.getLogger(__name__)


def load_update_subgraph(gds: GraphDataset):
    begin, end = 0, 0
    for _ in range(gds.num_epochs):
        begin, end = end, end + gds.batch_size
        choice = gds.randperm[begin:end]
        updage_subgraph = gds.data.edge_index[:, choice]
        num_div = gds.batch_size // 2
        yield updage_subgraph[:, :num_div], updage_subgraph[:, num_div:]

def update(
        upidx: int,
        update_subgraph_raw: Tuple[torch.Tensor, torch.Tensor],
        config: CommandLineArgument,
        gds: GraphDataset,
        csys: CacheSystem,
        task: Task,
        p: Profiler,
):
    logger.info(f'> #{upidx} update begin')

    # combine edges
    insert_edges, delete_edges = update_subgraph_raw
    insert_edges = insert_edges.to(config.device)
    delete_edges = delete_edges.to(config.device)

    # set update type with former insertion to 1, latter deletion to -1
    update_subgraph = torch.cat([insert_edges, delete_edges], dim=1)
    update_type = torch.cat([
        torch.ones(insert_edges.size(1), device=config.device),
        torch.ones(delete_edges.size(1), device=config.device).neg_(),
    ])

    p.begin(DEFAULT_CLK)
    affect_edges: Optional[torch.Tensor] = None
    nxt: Optional[LazyCache] = None
    for layer_idx in range(config.num_model_layers):
        p.record('load_edges', update_subgraph.size(1))
        dst, nxt = task.model.part_batch_compute(
            layer_idx=layer_idx,
            update_subgraph=update_subgraph,
            update_type=update_type,
            now_embd_cache=nxt,
            gds=gds,
            csys=csys,
            p=p,
        )

        if config.num_model_layers - layer_idx  > 1:
            affect_edges = gds.pma.affected_edges(dst)
            del update_subgraph
            del update_type
            update_subgraph = torch.cat([insert_edges, delete_edges, affect_edges], dim=1)
            logger.info(f'affected_nodes #{upidx} L{layer_idx}: {update_subgraph[1].unique().size(0)}')
            update_type = torch.cat([
                torch.ones(insert_edges.size(1), device=config.device),
                torch.ones(delete_edges.size(1), device=config.device).neg_(),
                torch.zeros(affect_edges.size(1), device=config.device),
            ])
        else:
            p.begin(COMPUTE_CLK)
            embd_out = nxt.cache.to(device='cpu', non_blocking=False)
            p.record('transmission', p.dur(COMPUTE_CLK))
            p.record('store_embd', embd_out.size(0))

    p.begin(COMPUTE_CLK)
    
    p.record('transmission', p.dur(COMPUTE_CLK))
    

    p.record('total', p.dur(DEFAULT_CLK))
    p.record('affect_edges', update_subgraph.size(1))
    p.record('affect_nodes', update_subgraph.view(-1).unique().size(0))

    dkeys = [k for k in p.checkpoints.keys()]
    dkeys.sort()
    for k in dkeys:
        v = p.checkpoints[k]
        p.logger.info(f'{k}: {v}')
    p.checkpoints.clear()

    p.logger.info(f'< finish graph update computation #{upidx}')


if __name__ == '__main__':
    p = Profiler()
    config = CommandLineArgument()
    logger.debug(config)

    torch.cuda.set_device(config.device)
    resource.setrlimit(resource.RLIMIT_AS, (config.mem_limit, config.mem_limit))
    del config.mem_limit

    logger.info(f'begin dataset fetch')
    p.begin(DEFAULT_CLK)
    gds = GraphDataset(
        dataset=config.dataset,
        root=config.root,
        batch_size=config.bs_update,
        num_epochs=config.num_epochs,
        device=config.device
    )
    p.logger.info(f'finish dataset fetch in {p.now(DEFAULT_CLK)}s')

    updater = load_update_subgraph(gds)

    logger.info(f'begin framework initialization')
    p.begin(DEFAULT_CLK)
    csys = CacheSystem(
        config=config,
        gds=gds,
    )

    task = Task(
        model=config.model,
        num_features=gds.data.num_features,
        num_hiddens=config.num_hiddens,
        num_classes=gds.data.num_classes,
        num_model_layers=config.num_model_layers,
        device=config.device,
        bs_compute=config.bs_compute,
    )

    for i, update_subgraph_raw in enumerate(updater):
        update(
            upidx=i,
            update_subgraph_raw=update_subgraph_raw,
            config=config,
            gds=gds,
            csys=csys,
            task=task,
            p=p,
        )

    dkeys = [k for k in p.records.keys()]
    dkeys.sort()
    for k in dkeys:
        v = p.records[k]
        p.logger.info(f'{k}: {v}')