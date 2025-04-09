import numpy as np
import pandas as pd
import torch, os, random
import logging

random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
logging.basicConfig(
    filename='logs/deg.log',
    filemode='w',
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

from lib.cli import CommandLineArgument
from lib.graph_dataset import GraphDataset
from typing import Optional
from torch_geometric.utils import to_torch_csr_tensor

sums90 = [0, 0, 0]
sums80 = [0, 0, 0]


def ratio(p: float, dst: torch.Tensor):
    num_nodes = dst.size(0)
    num_sels = int(p * num_nodes)

    top_dst = dst[ num_sels:]
    affected_edges = gds.pma.affected_edges(top_dst)
    affected_edges = affected_edges.unique(dim=1)
    top = affected_edges.size(1)
    logger.info(f'top {100 * (1 - p):.2f}%: {top}')

    low_dst = dst[:num_sels ]
    affected_edges = gds.pma.affected_edges(low_dst)
    affected_edges = affected_edges.unique(dim=1)
    low = affected_edges.size(1)
    logger.info(f'low {100 * p:.2f}%: {low}')

    affected_edges = gds.pma.affected_edges(dst)
    affected_edges = affected_edges.unique(dim=1)
    total = affected_edges.size(1)
    logger.info(f'100%: {total}')

    return top, low, total


def graph_update_actual(
        insert_edges: torch.Tensor,
        after_deletion_edges: torch.Tensor,
):
    gds.data.edge_index = torch.cat([
        after_deletion_edges, insert_edges.cpu()
    ], dim=1)
    gds.pma.insert_edges_actual(insert_edges)


def statistic(
        upidx: int,
        cfg: CommandLineArgument,
        gds: GraphDataset,
):
    insert_edges, delete_edges, edge_index = gds.load_update(
        batch_size=cfg.bs_update,
        num_epochs=cfg.num_epochs,
        epoch=upidx,
    )

    insert_edges = insert_edges.to(cfg.device)
    delete_edges = delete_edges.to(cfg.device)
    delayed_edges = insert_edges.clone()

    affect_edges: Optional[torch.Tensor] = None

    ins_dst = insert_edges[1]
    del_dst = delete_edges[1]


    new_dst, inverse = torch.cat([del_dst, ins_dst]).unique(
        return_inverse=True
    )

    csr = to_torch_csr_tensor(
        edge_index=edge_index,
        size=gds.data.num_nodes,
    )

    out_degree = torch.diff(csr.crow_indices())
    del csr

    degrees: torch.Tensor = out_degree[new_dst.cpu()]
    degrees, indice = degrees.sort()
    new_dst = new_dst[indice.to(cfg.device)]

    x, y, z = ratio(0.9, new_dst)
    sums90[0] += x
    sums90[1] += y
    sums90[2] += z
    x, y, z = ratio(0.8, new_dst)
    sums80[0] += x
    sums80[1] += y
    sums80[2] += z

    graph_update_actual(
        insert_edges=delayed_edges, 
        after_deletion_edges=edge_index
    )


if __name__ == '__main__':
    cfg = CommandLineArgument()
    gds = GraphDataset(
        dataset=cfg.dataset,
        root=cfg.root,
        batch_size=cfg.bs_update,
        num_epochs=cfg.num_epochs,
        device=cfg.device
    )

    for i in range(cfg.num_epochs):
        statistic(
            upidx=i,
            cfg=cfg,
            gds=gds,
        )

    logger.info(torch.Tensor(sums90) / cfg.num_epochs) 
    logger.info(torch.Tensor(sums80) / cfg.num_epochs) 




