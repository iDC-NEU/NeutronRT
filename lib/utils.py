import os
import sys
import gzip
import time
import shutil
import psutil   # for system profile
import logging
import pathlib
import requests # for downloading file

import dgl
import torch
import numpy as np

# types
from dgl import DGLGraph
from dgl.data import AsNodePredDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

KB = 1024
MB = KB * 1024
GB = MB * 1024

PID = os.getpid()
PROCESS = psutil.Process(PID)

NUM_FEATURES = 128
NUM_CLASSES = 50

PERF_LOG = 100

# for debugging

def tsstr(t: torch.Tensor) -> str:
    if isinstance(t, torch.Tensor):
        return f'({t.dtype} x {t.shape} @ {t.device})'
    return f'{t}'


def free_ram() -> float:
    return PROCESS.memory_info().rss / GB

# for time profile

_clocker = {}

def clk_begin(clock_name: str):
    _clocker[clock_name] = time.perf_counter()

def clk_end(clock_name: str) -> float:
    return time.perf_counter() - _clocker.pop(clock_name)

# for dataset

class CustomNodePredDataset:
    def __init__(self, g: DGLGraph, num_classes: int):
        self.graph: list = [g]
        self.num_classes = num_classes
    
    def __getitem__(self, idx):
        assert idx == 0, 'this dataset has only one graph'
        return self.graph[idx]
    
    def __len__(self):
        return len(self.graph)


def dummy_feat(num_nodes: int, num_features: int):
    return torch.rand(
        (num_nodes, num_features),
        dtype=torch.float32,
    )


def dummy_label(num_nodes: int, num_classes: int):
    return torch.randint(num_classes, (num_nodes,))


def webget(url: str, outfile: pathlib.Path):
    resp = requests.get(url)
    resp.raise_for_status()

    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(outfile, 'wb') as fout:
            with gzip.open(resp.raw, 'rb') as fin:
                shutil.copyfileobj(fin, fout)


def check_file(file_path: pathlib.Path, cached_file_path: pathlib.Path, url: str):
    if not file_path.parent.exists() or not file_path.is_file():
        file_path.parent.mkdir(exist_ok=True)
        logging.info(f'{file_path} no found, try to find cached file or download from internet')
        if not cached_file_path.is_file():
            logging.info(f'no cached file found, try to download from {url}')
            webget(url, file_path)
        else:
            logging.info(f'cached file found, try to load from {cached_file_path}')
            with open(file_path, 'wb+') as fout:
                with gzip.open(cached_file_path, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)
    
    if not file_path.is_file():
        raise FileNotFoundError
    
    logging.info(f'file check successfully')


def create_dataset(edge_index: torch.Tensor, **kwargs):
    num_nodes = kwargs.get('num_nodes', edge_index.max().item())
    num_features = kwargs.get('num_features', NUM_FEATURES)
    num_classes = kwargs.get('num_classes', NUM_CLASSES)

    feat = kwargs.get('feat', dummy_feat(num_nodes, num_features))
    label = kwargs.get('label', dummy_label(num_nodes, num_classes))

    u, v = edge_index
    g = dgl.graph(
        data=(u, v),
        num_nodes=num_nodes,
    )
    g.ndata['feat'] = feat
    g.ndata['label'] = label

    return CustomNodePredDataset(g, num_classes)
    

def download_dataset(dataset: str, root: pathlib.Path) -> DGLGraph:
    if isinstance(root, str):
        root = pathlib.Path(root)

    graph_cache_path = root.joinpath(f'.cache/{dataset}.graph')
    if graph_cache_path.is_file():
        logging.info(f'loading data from {graph_cache_path}')
        g = torch.load(graph_cache_path, map_location='cpu')
        return g
    else:
        if dataset.startswith('ogbn-'):
            graph_dataset = AsNodePredDataset(
                DglNodePropPredDataset(dataset, root)
            )
        elif dataset == 'reddit':
            graph_dataset = RedditDataset(raw_dir=root)
        elif dataset == 'twitter':
            # https://snap.stanford.edu/data/twitter-2010.html
            url = 'https://snap.stanford.edu/data/twitter-2010.txt.gz'
        
            file_path = root.joinpath('twitter-2010/twitter-2010.txt')
            cached_file_path = root.joinpath('twitter-2010.txt.gz')
            check_file(file_path, cached_file_path, url)

            txt = np.loadtxt(file_path, dtype=np.int64)
            edge_index = torch.from_numpy(txt)
            del txt
            # reindex
            _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
            edge_index: torch.Tensor = mapping.reshape(2, -1).contiguous()
            del mapping

            graph_dataset = create_dataset(edge_index, num_nodes=4165_2230)

            del edge_index
        
        elif dataset == 'friendster':
            # https://snap.stanford.edu/data/com-Friendster.html
            url = 'https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz'
            
            file_path = root.joinpath('friendster/com-friendster.ungraph.txt')
            cached_file_path = root.joinpath('com-friendster.ungraph.txt.gz')
            check_file(file_path, cached_file_path, url)

            txt = np.loadtxt(file_path, dtype=np.int64)
            edge_index = torch.from_numpy(txt)
            del txt
            # reindex
            _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
            edge_index: torch.Tensor = mapping.reshape(2, -1).contiguous()
            del mapping

            graph_dataset = create_dataset(edge_index, num_nodes=6560_8366)
    
            del edge_index
        else:
            raise ValueError(f'unsupport dataset {dataset}')
        
    g: DGLGraph = graph_dataset[0]

    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)
    g.create_formats_()
    g.pin_memory_()

    torch.save(g, graph_cache_path)
    return g


def split_randperm(num_edges: int, bs_update: int, num_epochs: int) -> torch.Tensor:
    rdperm = torch.randperm(num_edges)
    sel_rdperm = rdperm[:bs_update * num_epochs]
    split_rdperm = sel_rdperm.reshape((num_epochs, bs_update))
    logging.info(f'split randperm: {tsstr(split_rdperm)}')
    return split_rdperm
    

def final_affect_nodes(g: DGLGraph, target_nodes: torch.Tensor, num_layers: int) -> torch.Tensor:
    if target_nodes.device != g.device:
        target_nodes = target_nodes.to(g.device)
    aff_map = torch.zeros(g.num_nodes(), device=g.device, dtype=torch.bool)
    aff_map[target_nodes] = True
    for _ in range(num_layers - 1):
        u, v = g.out_edges(target_nodes)
        aff_map[v] = True
    return torch.where(aff_map)[0]


def next_affect_nodes(g: DGLGraph, target_nodes: torch.Tensor) -> torch.Tensor:
    if target_nodes.device != g.device:
        target_nodes = target_nodes.to(g.device)
    u, v = g.out_edges(target_nodes)
    target_nodes = torch.cat((target_nodes, v)).unique()
    return target_nodes


def next_affect_edges(g: DGLGraph, target_edges: torch.Tensor) -> torch.Tensor:
    if target_edges.device != g.device:
        target_edges = target_edges.to(g.device)
    u, v = g.edges()
    target_nodes = v[target_edges].unique()
    eid = g.out_edges(target_nodes, form='eid')
    target_edges = torch.cat((target_edges, eid))
    return target_edges
