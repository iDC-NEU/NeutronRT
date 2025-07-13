import torch as th

from typing import List
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler
from torch.nn.parallel import DistributedDataParallel


class Basic():
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.collect_nodes = []
        self.collect_subgraphs = []


    def affect_analysis(self, g: DGLGraph, update: th.Tensor, samplers: List[NeighborSampler]): ...


    def layerwise_inference(self, g: DGLGraph, ddp: DistributedDataParallel, samplers: List[NeighborSampler], feats: th.Tensor): ...