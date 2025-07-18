import os
import toml
import time
import argparse
import numpy as np
import dgl
import dgl.nn as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.dataset import load_dataset
from lib.graph import random_eidperm
from lib.methods import load_method, Method, Incremental
from lib.models import GNNModel

# types
from dgl.dataloading import DataLoader, NeighborSampler
from torch.nn.parallel import DistributedDataParallel
from ogb.nodeproppred import DglNodePropPredDataset
from typing import Any, List, Dict
from dgl.heterograph import DGLBlock


def build_embd_cache(
        num_layers: int, 
        num_nodes: int, 
        num_hiddens: int, 
        device: th.device, 
        is_pinned: bool, 
        is_shared: bool, 
        dtype: th.dtype
):
    hcaches = []
    for _ in range(1, num_layers):
        dummy = th.rand(
            (num_nodes, num_hiddens),
            dtype=dtype,
            device=device,
            pin_memory=is_pinned,
        )
        if is_shared:
            dummy.share_memory_()
        hcaches.append(dummy)
    return hcaches


def build_aggr_cache(
        model: str,
        num_layers: int,
        num_nodes: int, 
        num_features: int, 
        num_hiddens: int, 
        device: th.device,
        is_pinned: bool,
        is_shared: bool,
        dtype: th.dtype
):
    acaches = []
    for layer_idx in range(num_layers):
        if model == 'sage':
            num_dim = num_hiddens
        else:
            num_dim = num_features if layer_idx == 0 else num_hiddens

        dummy = th.rand(
            (num_nodes, num_dim),
            dtype=dtype,
            device=device,
            pin_memory=is_pinned,
        )
        if is_shared:
            dummy.share_memory_()
        acaches.append(dummy)
    return acaches


def handle_update(
        rank: int,
        world_size: int,
        g: dgl.DGLGraph,
        feats: th.Tensor,
        ddp: DistributedDataParallel,
        m: Method,
        samplers: List[NeighborSampler],
):
    for idx, layer_nodes in enumerate(m.collect_nodes):
        for layer_idx, nodes in enumerate(layer_nodes):
            m.collect_nodes[idx][layer_idx] = nodes[rank::world_size]

    if isinstance(m, Incremental) and ddp.module.use_softmax:
        for idx, layer_nodes in enumerate(m.collect_recomputed):
            for layer_idx, nodes in enumerate(layer_nodes):
                m.collect_recomputed[idx][layer_idx] = nodes[rank::world_size]

    th.cuda.synchronize()
    total_time = time.perf_counter()

    with th.no_grad():
        compt_time = m.layerwise_inference(g, ddp, samplers, feats)

    th.cuda.synchronize()
    total_time = time.perf_counter() - total_time
    st_time = total_time - compt_time

    print(f"Rank {rank}: Inference finished.")
    print(f"Rank {rank}: Time> Total: {total_time:.4f}, Compute: {compt_time:.4f}, Sample+Transfer: {st_time:.4f},")

    return compt_time, st_time


def run(
        rank: int, 
        world_size: int, 
        config: Dict[str, Any], 
        g: dgl.DGLGraph,
        feats: th.Tensor, 
        labels: th.Tensor,
        method: Method,
        samplers: List[NeighborSampler],
    ):
    print(f"Starting process {rank}/{world_size}...")
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    th.cuda.set_device(rank)

    model = GNNModel(
        model_name=config['model']['name'],
        num_features=config['model']['num_features'],
        num_hiddens=config['model']['num_hiddens'],
        num_classes=config['model']['num_classes'],
        num_layers=config['model']['num_layers'],
        use_softmax=config['model']['use_softmax'],
        **config['model']['kwargs']
    ).to(rank)
    
    model = DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    print(f'Rank {rank}: Begin> {time.perf_counter():.4f}')
    with g.local_scope():
        records = handle_update(rank, world_size, g, feats, model, method, samplers)
    th.cuda.synchronize()
    print(f'Rank {rank}: End> {time.perf_counter():.4f}')
            
    dist.destroy_process_group()
     
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='config/default.toml',
        help='set the config file of the server',
    )
    args = parser.parse_args()
    config = toml.load(args.config)

    print("--- Configuration ---")
    print(config)
    print("---------------------")
    
    # 设置环境变量
    for k, v in config['env'].items():
        os.environ[k] = str(v)

    th.manual_seed(0)
    if th.cuda.is_available():
        th.cuda.manual_seed(0)
        th.cuda.manual_seed_all(0)

    print("Loading dataset...")
    dataset = load_dataset(config['dataset']['name'], config['file']['root'])
    graph: dgl.DGLGraph = dataset[0]
    graph.create_formats_()

    updates = random_eidperm(graph.num_edges(), config['parameter']['bs_update'], config['parameter']['num_snapshots'])

    labels: th.Tensor = graph.ndata['label'].squeeze()
    features: th.Tensor = graph.ndata['feat']

    print(f"Graph: {graph}")
    print(f"Features: {features.shape}")
    print(f"Labels: {labels.shape}")

    config['model']['num_features'] = features.shape[1]
    config['model']['num_classes'] = int(labels.max().item() + 1)

    samplers = [NeighborSampler([f]) for f in config['model']['fanouts']]
    
    if not config['dataset']['in_memory']:
        print("Moving graph to shared memory...")
        graph = graph.shared_memory(config['shared']['shm_name'])
        labels.share_memory_()
        features.share_memory_()
    else:
        print("Moving graph to device side...")
        graph = graph.to('cuda')
        labels = labels.cuda()
        features = features.cuda()

    hcaches = build_embd_cache(
        config['model']['num_layers'],
        graph.num_nodes(),
        config['model']['num_hiddens'],
        graph.device,
        graph.is_pinned(),
        not config['dataset']['in_memory'],
        features.dtype
    )
    hcaches.insert(0, features)
    print(f'h caches: {hcaches}')

    acaches = build_aggr_cache(
        config['model']['name'],
        config['model']['num_layers'],
        graph.num_nodes(),
        config['model']['num_features'],
        config['model']['num_hiddens'],
        graph.device,
        graph.is_pinned(),
        not config['dataset']['in_memory'],
        features.dtype
    )
    print(f'a caches: {acaches}')

    feats = (hcaches, acaches)

    world_size = config['dist']['world_size']
    print(f"Spawning {world_size} processes...")

    method: Method = load_method(config['parameter']['method'])(config['model']['num_layers'])
    th.cuda.synchronize()
    affxt_time = time.perf_counter()
    method.affect_analysis(graph, updates, samplers, config['model']['use_softmax'])
    th.cuda.synchronize()
    affxt_time = time.perf_counter() - affxt_time
    print(f"Main: Time> Affect: {affxt_time:.4f},")

    parallel_time = time.perf_counter()
    mp.spawn(run, args=(world_size, config, graph, feats, labels, method, samplers), nprocs=world_size, join=True)
    parallel_time = time.perf_counter() - parallel_time
    print(f"Inference completed.")