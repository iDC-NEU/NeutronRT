import torch
import pathlib
import logging
import numpy as np
from dgl.data import AsNodePredDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset

from ..graph.metagraph import MetaGraph

from dgl import DGLGraph
from typing import Optional

def load_dataset(dataset: str, root: str, cache_dir: Optional[str] = None) -> MetaGraph:
    NUM_FEATURES = 128
    NUM_CLASSES = 50

    root_dir = pathlib.Path(root)

    if cache_dir is None:
        cache_dir: pathlib.Path = root_dir.joinpath('.cache')
    else:
        cache_dir = pathlib.Path(cache_dir)

    graph_cache: pathlib.Path = cache_dir.joinpath(f'{dataset}.graph')

    if cache_dir.is_dir() and graph_cache.is_dir():
        logging.info(f'loading graph data from {graph_cache}')
        g: MetaGraph = torch.load(graph_cache, map_location='cpu')
    else:
        if dataset.startswith('ogbn-'):
            graph_dataset = AsNodePredDataset(
                DglNodePropPredDataset(dataset, root)
            )
            g: DGLGraph = graph_dataset[0]
            num_classes = graph_dataset.num_classes

            g = MetaGraph(g, num_classes=num_classes)

        elif dataset == 'reddit':
            graph_dataset = RedditDataset(raw_dir=root)
            g: DGLGraph = graph_dataset[0]
            num_classes = graph_dataset.num_classes

            g = MetaGraph(g, num_classes=num_classes)
    
        elif dataset == 'twitter':
            # https://snap.stanford.edu/data/twitter-2010.html
            file_path = root_dir.joinpath('twitter-2010/twitter-2010.txt')

            txt = np.loadtxt(file_path, dtype=np.int64)
            edge_index = torch.from_numpy(txt)
            # reindex
            _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
            edge_index: torch.Tensor = mapping.reshape(2, -1).contiguous()

            g = MetaGraph(edge_index, num_nodes=4165_2230, num_classes=NUM_CLASSES, num_features=NUM_FEATURES)
        
        elif dataset == 'friendster':
            # https://snap.stanford.edu/data/com-Friendster.html
            file_path = root_dir.joinpath('friendster/com-friendster.ungraph.txt')

            txt = np.loadtxt(file_path, dtype=np.int64)
            edge_index = torch.from_numpy(txt)
            # reindex
            _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
            edge_index: torch.Tensor = mapping.reshape(2, -1).contiguous()

            g = MetaGraph(edge_index, num_nodes=6560_8366, num_classes=NUM_CLASSES, num_features=NUM_FEATURES)
        else:
            raise ValueError(f'unsupport dataset {dataset}')

    if cache_dir.is_dir():
        torch.save(g, graph_cache)
    return g


        