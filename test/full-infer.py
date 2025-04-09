import logging, pathlib, os, time, random, resource, torch, scipy
import torch.utils.data
import torch.utils.data.dataloader

# system initialization arguments
random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
logging.basicConfig(
    filename='logs/full.log',
    filemode='w',
    level=logging.DEBUG,
)

import torch.utils
from torch_geometric.utils import degree, k_hop_subgraph
from torch_geometric.loader import NeighborLoader, DataLoader

from typing import Optional, List

from lib.graph_dataset import GraphDataset
from lib.cli import CommandLineArgument
from lib.profiler import Profiler
from lib.cache_system import CacheSystem
from lib.task import Task

# clocks for profiler
DEFAULT_CLK = 1
PART_COMPUTE_CLK = 2

# global __main__ logger
logger = logging.getLogger(__name__)


def tensor_text(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        return f'{t.dtype}@{t.device} {t.shape}'
    else:
        return f'{t}'
    

def cache_initialization(
        config: CommandLineArgument,
        gds: GraphDataset,
        csys: CacheSystem,
        task: Task,
):
    # for large graphs
    # do neighbour sample to avoid OOM
    loader = NeighborLoader(
        edge_index=gds.data.edge_index,
        node_idx=torch.arange(gds.data.num_nodes),
        sizes=[-1],
        batch_size=config.bs_init,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
    )

    for layer_idx in range(config.num_model_layers):
        embd_cache = csys.embd_cache[layer_idx]
        aggr_cache = csys.aggr_cache[layer_idx]
        
        if layer_idx + 1 < config.num_model_layers:
            next_embd_cache = csys.embd_cache[layer_idx + 1]
        else:
            next_embd_cache = None

        for batch_size, n_id, adjs in loader:
            x = embd_cache.host[n_id].to(config.device)
            edge_index = adjs[0].to(config.device)
            embd_out, aggr_out = task.model.full_compute(
                layer_idx=layer_idx,
                x=x,
                edge_index=edge_index,
                batch_size=batch_size,
            )

            nodes = n_id[:batch_size]

            # embedding
            if next_embd_cache is not None:
                next_embd_cache.host[nodes].copy_(embd_out, non_blocking=True)

            # aggregation
            aggr_cache.host[nodes].copy_(aggr_out, non_blocking=True)


def update_embd_cache(
        layer_idx: int,
        csys: CacheSystem,
        dst_index: torch.Tensor,
        new_embd: torch.Tensor
):
    csys.update_embd(
        layer_idx=layer_idx,
        targets=dst_index.cpu(),
        new_embd=new_embd,
    )


def affected_nodes(dst: torch.Tensor):
    nodes = gds.pma.affected_nodes(dst)
    return nodes


def graph_update_actual(
        insert_edges: torch.Tensor,
        after_deletion_edges: torch.Tensor,
):
    gds.data.edge_index = torch.cat([
        after_deletion_edges, insert_edges.cpu()
    ], dim=1)
    gds.pma.insert_edges_actual(insert_edges)


def update(
        upidx: int,
        config: CommandLineArgument,
        gds: GraphDataset,
        csys: CacheSystem,
        task: Task,
        p: Profiler,
):
    p.begin(DEFAULT_CLK)
    insert_edges, delete_edges, edge_index = gds.load_update(
        batch_size=config.bs_update,
        num_epochs=config.num_epochs,
        epoch=upidx,
    )

    # combine edges
    insert_edges = insert_edges.to(config.device)
    delete_edges = delete_edges.to(config.device)

    graph_update_actual(insert_edges, edge_index)

    # set update type with former insertion to 1, latter deletion to -1
    p.record('update.graph_update', p.dur(DEFAULT_CLK))
    p.logger.info(f'finish graph update #{upidx}')

    affect_nodes: Optional[torch.Tensor] = None
    for layer_idx in range(config.num_model_layers):
        # delete and affected edges use old embedding caches
        if affect_nodes is not None:
            dst = torch.unique(torch.cat([dst, affect_nodes]))
        else:
            dst = torch.cat([insert_edges[1], delete_edges[1]]).unique()

        p.record('update.destination', p.dur(DEFAULT_CLK))

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=dst,
            num_hops=1,
            edge_index=gds.data.edge_index,
            relabel_nodes=True,
            num_nodes=gds.data.num_nodes,
            flow='source_to_target',
            directed=True,
        )

        logger.info(f'edge_index: {edge_index.shape}')
        
        p.record('update.neighbour_search', p.dur(DEFAULT_CLK))

        x = csys.fetch_embd(
            layer_idx=layer_idx,
            targets=subset,
            device=config.device,
        )

        p.record('update.fetch_cache', p.dur(DEFAULT_CLK))

        y, _ = task.model.full_compute(
            layer_idx=layer_idx,
            x=x,
            edge_index=edge_index.to(config.device),
        )

        p.record('update.full_compute', p.dur(DEFAULT_CLK))
        
        if layer_idx + 1 < config.num_model_layers:
            csys.update_embd(
                layer_idx=layer_idx + 1,
                targets=dst,
                new_embd=y[mapping.to(config.device)],
            )

        p.record('update.update_cache', p.dur(DEFAULT_CLK))

        if layer_idx + 1 < config.num_model_layers:
            affect_nodes = affected_nodes(dst)

        p.record('update.fetch_affected_edges', p.dur(DEFAULT_CLK))


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

    logger.info(f'begin framework initialization')
    p.begin(DEFAULT_CLK)
    csys = CacheSystem(
        config=config,
        gds=gds,
    )
    del config.embd_host_ratio
    del config.embd_cuda_ratio
    del config.aggr_host_ratio
    del config.aggr_cuda_ratio

    task = Task(
        model=config.model,
        num_features=gds.data.num_features,
        num_hiddens=config.num_hiddens,
        num_classes=gds.data.num_classes,
        num_model_layers=config.num_model_layers,
    )

    # move model to device side
    task.model.to(config.device)

    # logger.info(f'begin cache initialization')
    # cache_initialization(
    #     config=config,
    #     gds=gds,
    #     csys=csys,
    #     task=task,
    # )
    # p.logger.info(f'finish framework initialization in {p.now(DEFAULT_CLK)}s')

    logger.debug(config)

    for i in range(config.num_epochs):
        update(
            upidx=i,
            config=config,
            gds=gds,
            csys=csys,
            task=task,
            p=p,
        )

    for k in p.records.keys():
        v = p.records[k]
        p.logger.info(f'{k}: {v}s')

