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
    filename='logs/compute-3-eco.log',
    filemode='w',
    level=logging.INFO,
)

import numpy as np
import ogb.nodeproppred as ogbn
import torch_geometric.datasets as pygds
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from typing import Optional, List

from lib.cli import CommandLineArgument
from lib.profiler import Profiler
from lib.task import Task
from lib.pma_graph_cpp import PMAGraph
from lib.cache_system import LayerCache
from lib.model import compute_batch_range

# clocks for profiler
DEFAULT_CLK = 1
COMPUTE_CLK = 2

# global __main__ logger
logger = logging.getLogger(__name__)

p = Profiler()
config = CommandLineArgument()
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


embd_cache: List[LayerCache] = []
embd_host_ratio = [r if r <= 1.0 else min(r / data.num_nodes, 1.0) for r in config.embd_host_ratio]
embd_cuda_ratio = [r if r <= 1.0 else min(r / data.num_nodes, 1.0) for r in config.embd_cuda_ratio]
# assuming host can always hold all feat
if embd_cuda_ratio[0] < 1:
    embd_in_memory = False
    host_embd = data.x.pin_memory()
    embd_cuda_key = None
    cuda_embd = None
else:
    embd_in_memory = True
    host_embd = None
    embd_cuda_key = None
    cuda_embd = data.x.to(config.device, non_blocking=True)

embd_cache.append(
    LayerCache(
        num_dim=data.x.size(1),
        layer_idx=0,
        cache_type='embd',
        cuda_cache=cuda_embd,
        cuda_key=embd_cuda_key,
        host_cache=host_embd,
    )
)

for layer_idx in range(config.num_model_layers):
    num_dim =  config.num_hiddens if layer_idx != config.num_model_layers - 1 else data.num_classes

    if config.num_model_layers - layer_idx > 1:
        if embd_cuda_ratio[layer_idx] < 1:
            embd_in_memory = False
            host_embd = torch.zeros((data.num_nodes, num_dim), dtype=data.x.dtype, pin_memory=True)
            embd_cuda_key = None
            cuda_embd = None
        else:
            embd_in_memory = True
            host_embd = None
            embd_cuda_key = None

            # create cuda embedding cache
            cuda_embd = torch.zeros((data.num_nodes, num_dim), dtype=data.x.dtype, device=config.device)

        embd_cache.append(
            LayerCache(
                num_dim=num_dim,
                layer_idx=layer_idx + 1,
                cache_type='embd',
                cuda_cache=cuda_embd,
                cuda_key=embd_cuda_key,
                host_cache=host_embd,
            )
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
    nodes_idx = dst.cpu()

    p.begin(DEFAULT_CLK)
    for layer_idx in range(config.num_model_layers):
        nloader = NeighborLoader(
            data=data,
            num_neighbors=[-1],
            input_nodes=dst,
            batch_size=config.bs_compute,
            num_workers=config.num_workers,
        )
    
        for batch in nloader:
            p.record('load_nodes', batch.n_id.size(0))
            p.record('load_edges', batch.e_id.size(0))

            nodes_idx = torch.cat([nodes_idx, batch.n_id]).unique()
            edges_idx = torch.cat([edges_idx, batch.e_id]).unique()

            # initialize x with feat
            p.begin(COMPUTE_CLK)
            x = embd_cache[layer_idx].fetch(batch.n_id, device=config.device)
            p.record('transmission', p.dur(COMPUTE_CLK))
            p.record('load_embd', batch.n_id.size(0))

            ld_src, ld_dst = batch.edge_index.to(config.device)
            ld_dst, ld_indice, dst_in_degrees = ld_dst.unique(
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            ld_indice, reorder = ld_indice.sort()
            reorder = reorder.cpu()

            for begin, end, cbegin, cend in compute_batch_range(config.bs_compute, dst_in_degrees):
                p.record('nbatches', 1)
                batch_dst = ld_dst[begin:end]
                batch_edge_index = batch.edge_index[:, reorder[cbegin:cend]].to(config.device)

                p.begin(COMPUTE_CLK)
                batch_x_out = task.model.full_compute(
                    layer_idx=layer_idx,
                    x=x,
                    edge_index=batch_edge_index,
                )
                p.record('computation', p.dur(COMPUTE_CLK))
    
                if config.num_model_layers - layer_idx > 1:
                    # ignore accuracy for perf test
                    p.begin(COMPUTE_CLK)
                    embd_cache[layer_idx + 1].update(batch_dst, batch_x_out[:batch_dst.size(0)], non_blocking=True)
                    p.record('cache.update', p.dur(COMPUTE_CLK))
                    p.record('store_embd', batch_dst.size(0))
                else:
                    p.begin(COMPUTE_CLK)
                    embd_out = batch_x_out.to(device='cpu', non_blocking=False)
                    p.record('transmission', p.dur(COMPUTE_CLK))
                    p.record('store_embd', embd_out.size(0))

    
            if config.num_model_layers - layer_idx > 1:
                dst = dst.to(config.device)
                affect_nodes = pma.affected_nodes(dst)
                dst = torch.cat([dst, affect_nodes]).unique()
                logger.info(f'affected_nodes #{upidx} L{layer_idx}: {dst.size(0)}')

    p.record('total', p.dur(DEFAULT_CLK))
    
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



