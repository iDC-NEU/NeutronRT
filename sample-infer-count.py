import logging, os, random, sys, torch, numpy

# system initialization arguments
random.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)

# set environment variable
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['NUMEXPR_MAX_THREADS'] = '40'

import numpy as np
import ogb.nodeproppred as ogbn
import torch_geometric.datasets as pygds
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from typing import Optional

from lib.cli import CommandLineArgument
from lib.profiler import Profiler
from lib.task import Task
from lib.pma_graph_cpp import PMAGraph

# global __main__ logger
p = Profiler()
config = CommandLineArgument()

# set logger file
if config.sample[0] != -1:
    logging.basicConfig(
        filename='logs/count-2-smp.log',
        filemode='w',
        level=logging.INFO,
    )
else:
    logging.basicConfig(
        filename='logs/count-1-full.log',
        filemode='w',
        level=logging.INFO,
    )

logger = logging.getLogger(__name__)
logger.info(config)

if config.dataset.startswith('ogbn-'):
    gdata = ogbn.PygNodePropPredDataset(name=config.dataset, root=config.root)

    data: Data = gdata.data
    data.num_classes = gdata.num_classes
elif config.dataset.find('reddit') != -1:
    gdata = pygds.Reddit(root=config.root)

    data: Data = gdata.data
    data.num_classes = gdata.num_classes
elif config.dataset == 'twitter':
    # https://snap.stanford.edu/data/twitter-2010.html

    file_path = config.root.joinpath('twitter-2010/twitter-2010.txt')
    logger.info(f'loading data from {file_path}')
    edge_index = torch.from_numpy(np.loadtxt(file_path, dtype=np.int64))
    edge_index.transpose_(0, 1)
    edge_index = edge_index.contiguous()
    logger.info(f'loading finished')

    # edge_index = to_torch_coo_tensor(edge_index)
    num_nodes = 4165_2230
    x = torch.rand(
        (num_nodes, 128),
        dtype=torch.float32,
        pin_memory=True
    )
    data = Data(x=x, edge_index=edge_index)
    data.num_classes = 50
elif config.dataset == 'com-friendster':
    # https://snap.stanford.edu/data/com-Friendster.html
    
    file_path = config.root.joinpath('com-friendster/com-friendster.ungraph.txt')
    logger.info(f'loading data from {file_path}')
    edge_index = torch.from_numpy(np.loadtxt(file_path, dtype=np.int64))
    _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
    edge_index: torch.Tensor = mapping.reshape(2, -1)
    edge_index = edge_index.contiguous()
    logger.info(f'loading finished')

    # edge_index = to_torch_coo_tensor(edge_index)
    num_nodes = 6560_8366
    x = torch.rand(
        (num_nodes, 128),
        dtype=torch.float32,
        pin_memory=True
    )
    data = Data(x=x, edge_index=edge_index)
    data.num_classes = 50
else:
    raise ValueError(f'unsupport dataset {config.dataset}')


logger.info(data)

if config.dataset in ['ogbn-arxiv', 'reddit', 'ogbn-products']:
    data = data.to(config.device)
    config.bs_update = data.num_edges // 1000
    logger.info(f'real batch size update: {config.bs_update}')

randperm = torch.randperm(data.edge_index.size(1))
randperm = randperm[:config.num_epochs * config.bs_update]

task = Task(
    model=config.model,
    num_features=data.num_features,
    num_hiddens=config.num_hiddens,
    num_classes=data.num_classes,
    num_model_layers=config.num_model_layers,
    device=config.device,
    bs_compute=config.bs_compute,
)


pma = PMAGraph(
    num_nodes=data.num_nodes,
    edge_index=data.edge_index,
    device=config.device,
    max_num_edges=data.num_edges,
)

begin, end = 0, 0
for upidx in range(config.num_epochs):
    logger.info(f'> #{upidx} update begin')
    begin, end = end, end + config.bs_update

    edges_idx = randperm[begin:end]
    dst = data.edge_index[1][edges_idx].unique()
    nodes_idx = dst

    affect_nodes: Optional[torch.Tensor] = None
    for layer_idx in range(config.num_model_layers - 1):
        affect_nodes = pma.affected_nodes(dst)
        dst = torch.cat([dst, affect_nodes]).unique()
    p.record('store_nodes', dst.size(0))

    nloader = NeighborLoader(
        data=data,
        num_neighbors=config.sample[::-1],
        input_nodes=dst,
        batch_size=config.bs_compute,
        num_workers=config.num_workers,
    )

    for batch in nloader:
        batch = batch.to(config.device)
        p.record('nbatches', 1)
        p.record('load_nodes', batch.n_id.size(0))
        p.record('load_edges', batch.e_id.size(0))
        nodes_idx = torch.cat([nodes_idx, batch.n_id]).unique()
        edges_idx = torch.cat([edges_idx, batch.e_id]).unique()
    
    p.record('affect_nodes', nodes_idx.size(0))
    p.record('affect_edges', edges_idx.size(0))

    dkeys = [k for k in p.checkpoints.keys()]
    dkeys.sort()
    for k in dkeys:
        v = p.checkpoints[k]
        p.logger.info(f'{k}: {v}')
    p.checkpoints.clear()
    logger.info(f'< #{upidx} update finished')


dkeys = [k for k in p.records.keys()]
dkeys.sort()
for k in dkeys:
    v = p.records[k]
    p.logger.info(f'{k}: {v}')



