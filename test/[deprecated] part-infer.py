import logging, pathlib, os, time, random, resource, torch, scipy
import torch.utils.data
import torch.utils.data.dataloader

# system initialization arguments
random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['PYTHONPATH'] = '..'
logging.basicConfig(
    filename='logs/part.log',
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


def part_aggregation(
        layer_idx: int,
        edge_index: torch.Tensor,
        type: int,
        num_dims: int,
        csys: CacheSystem,
        task: Task,
        p: Profiler,
):
    p.begin(PART_COMPUTE_CLK)

    src, dst = edge_index
    dst, indice, counts = dst.unique(
            sorted=True,
            return_inverse=True,
            return_counts=True,
    )
    scatter_index = indice.view(-1, 1).repeat(1, num_dims)

    dst_cpu: torch.Tensor = dst.cpu()

    p.record('part_aggregation.misc', p.dur(PART_COMPUTE_CLK))

    src_embd = csys.fetch_embd(
        layer_idx=layer_idx,
        targets=src,
        device=edge_index.device
    )
    
    p.record('part_aggregation.fetch_cache', p.dur(PART_COMPUTE_CLK))

    if type == 1:
        # insert edges
        pass
    elif type == -1:
        # delete edges
        src_embd.neg_()
        counts.neg_()
    else:
        raise ValueError
    
    p.record('part_aggregation.misc', p.dur(PART_COMPUTE_CLK))
    
    part_aggr = torch.zeros(
        (dst.size(0), num_dims),
        dtype=src_embd.dtype,
        device=edge_index.device,
    ).scatter_reduce_(0, scatter_index, src_embd, reduce=config.aggr_method)

    p.record('part_aggregation.scatter_reduce', p.dur(PART_COMPUTE_CLK))

    old_aggr = csys.fetch_aggr(
        layer_idx=layer_idx,
        targets=dst_cpu,
        device=config.device
    )

    p.record('part_aggregation.fetch_cache', p.dur(PART_COMPUTE_CLK))

    part_aggr = csys.part_aggregation(
        targets=dst,
        part_aggr=part_aggr,
        old_aggr=old_aggr,
        in_degree=counts.to(config.device),
    )

    p.record('part_aggregation.combine_aggregation', p.dur(PART_COMPUTE_CLK))

    csys.update_aggr(
        layer_idx=layer_idx,
        targets=dst_cpu,
        new_aggr=part_aggr,
        non_blocking=True,
    )

    p.record('part_aggregation.update_cache', p.dur(PART_COMPUTE_CLK))

    x = csys.fetch_embd(
        layer_idx=layer_idx,
        targets=dst_cpu,
        device=config.device,
    )

    p.record('part_aggregation.fetch_cache', p.dur(PART_COMPUTE_CLK))

    embd = task.model.part_compute(
        layer_idx=layer_idx,
        x=x,
        edge_index=edge_index,
        aggr_out=part_aggr,
    )

    p.record('part_aggregation.neural_network', p.dur(PART_COMPUTE_CLK))

    return dst, embd


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


def affected_graph(dst: torch.Tensor):
    affected_edges = gds.pma.affected_edges(dst)
    affected_edges = affected_edges.unique(dim=1)
    logger.debug(f'affected edges: {dst.shape} {affected_edges.shape}')
    return affected_edges


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
    delayed_edges = insert_edges.clone()

    # set update type with former insertion to 1, latter deletion to -1
    p.record('update.graph_update', p.dur(DEFAULT_CLK))
    p.logger.info(f'finish graph update #{upidx}')

    affect_edges: Optional[torch.Tensor] = None
    for layer_idx in range(config.num_model_layers):
        num_dims = gds.data.num_features if layer_idx == 0 else config.num_hiddens

        # delete and affected edges use old embedding caches
        if affect_edges is not None:
            delete_edges = torch.unique(torch.cat(
                [delete_edges, affect_edges],
                dim=1,
            ), dim=1)

            insert_edges = torch.unique(torch.cat(
                [insert_edges, affect_edges],
                dim=1,
            ), dim=1)
        
        p.record('update.insert_affected_edges', p.dur(DEFAULT_CLK))

        del_dst, del_embd = part_aggregation(
            layer_idx=layer_idx,
            edge_index=delete_edges,
            type=-1,
            num_dims=num_dims,
            csys=csys,
            task=task,
            p=p,
        )
        
        p.record('update.part_aggregation', p.dur(DEFAULT_CLK))

        if affect_edges is not None:
            update_embd_cache(
                layer_idx=layer_idx,
                csys=csys,
                dst_index=new_dst,
                new_embd=new_embd,
            )
        
        p.record('update.update_embedding', p.dur(DEFAULT_CLK))

        ins_dst, ins_embd = part_aggregation(
            layer_idx=layer_idx,
            edge_index=insert_edges,
            type=1,
            num_dims=num_dims,
            csys=csys,
            task=task,
            p=p,
        )

        p.record('update.part_aggregation', p.dur(DEFAULT_CLK))
        
        if layer_idx + 1 < config.num_model_layers:
            new_dst, inverse = torch.cat([del_dst, ins_dst]).unique(
                return_inverse=True
            )
            new_embd = torch.zeros(
                (new_dst.size(0), num_dims),
                dtype=gds.data.x.dtype,
                device=config.device,
            )

            del_inverse = inverse[:del_dst.size(0)]
            new_embd[del_inverse].copy_(del_embd)

            ins_inverse = inverse[del_dst.size(0):]
            new_embd[ins_inverse].copy_(ins_embd)

            affect_edges = affected_graph(new_dst)

        p.record('update.fetch_affected_edges', p.dur(DEFAULT_CLK))

    graph_update_actual(delayed_edges, edge_index)
    p.record('update.graph_update', p.dur(DEFAULT_CLK))


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

