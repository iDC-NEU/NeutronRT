import torch, logging, os, time, datetime
import torch_geometric
import ogb.nodeproppred as ogbn
import torch_geometric.datasets as pygds
import numpy as np
import pandas as pd

from pathlib import Path as P
from torch_scatter import scatter

from torch_geometric.data import Data

from .pma_graph_cpp import PMAGraph
from .utils import free_ram, tensor_text

from typing import Optional


class GraphDataset():
    def __init__(
            self,
            dataset: str,
            root: P,
            batch_size: int,
            num_epochs: int,
            device: torch.device
    ):
        # dataset info
        self.dataset = dataset
        self.root = root
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # init logger
        self.logger = logging.getLogger(name='dataset')

        if self.dataset in ['ogbn-arxiv', 'ogbn-products', 'reddit']:
            self.in_memory = True
            self.logger.info(f'load to cuda')
        else:
            self.in_memory = False
            self.logger.info(f'load to cpu')

        self.logger.info(f'{free_ram():.2f}GB already allocated before graph data import')

        self.download_dataset()

        self.logger.info(f'{free_ram():.2f}GB already allocated after dataset downloading')

        self.pma = PMAGraph(
            num_nodes=self.data.num_nodes,
            edge_index=self.data.edge_index,
            device=device,
            max_num_edges=self.data.num_edges,
        )

        self.logger.debug(f'{free_ram():.2f}GB already allocated after pma graph initialization')
        self.logger.debug(f'{self.data}')
        self.logger.debug(f'real batch size update: {self.batch_size}')


    def download_dataset(self):
        if self.dataset.startswith('ogbn-'):
            gdata = ogbn.PygNodePropPredDataset(name=self.dataset, root=self.root)
            data: Data = gdata.data
            data.num_classes = gdata.num_classes
        elif self.dataset.find('reddit') != -1:
            gdata = pygds.Reddit(root=self.root)
            data: Data = gdata.data
            data.num_classes = gdata.num_classes
        elif self.dataset == 'twitter':
            # https://snap.stanford.edu/data/twitter-2010.html

            file_path = self.root.joinpath('twitter-2010/twitter-2010.txt')
            self.logger.info(f'loading data from {file_path}')
            edge_index = torch.from_numpy(np.loadtxt(file_path, dtype=np.int64))
            edge_index.transpose_(0, 1)
            edge_index = edge_index.contiguous()
            self.logger.info(f'loading finished')

            # edge_index = to_torch_coo_tensor(edge_index)
            num_nodes = 4165_2230
            x = torch.rand(
                (num_nodes, 128),
                dtype=torch.float32,
                pin_memory=True
            )
            data = Data(x=x, edge_index=edge_index)
            data.num_classes = 50

        elif self.dataset == 'com-friendster':
            # https://snap.stanford.edu/data/com-Friendster.html
            
            file_path = self.root.joinpath('com-friendster/com-friendster.ungraph.txt')
            self.logger.info(f'loading data from {file_path}')
            edge_index = torch.from_numpy(np.loadtxt(file_path, dtype=np.int64))
            _, mapping = edge_index.transpose_(0, 1).reshape(-1).unique(return_inverse=True)
            edge_index: torch.Tensor = mapping.reshape(2, -1)
            edge_index = edge_index.contiguous()
            self.logger.info(f'loading finished')

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
            raise ValueError(f'unsupport dataset {self.dataset}')
        
        self.data = data

        self.logger.debug(f'edge_index shape: {self.data.edge_index.shape}')
        self.logger.debug(f'{free_ram():.2f}GB already allocated after data fetching')

        if self.in_memory:
            # self.data = self.data.to(self.device)
            self.batch_size = self.data.num_edges // 1000
        self.randperm = torch.randperm(data.edge_index.size(1))[:self.num_epochs * self.batch_size]




            
            
