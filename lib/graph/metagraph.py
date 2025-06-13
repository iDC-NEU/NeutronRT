import dgl
import torch

from dgl import DGLGraph
from torch import Tensor
from typing import Optional

class MetaGraph():
    def __init__(
            self,
            g,
            *,
            num_nodes: Optional[int] = None,
            num_features: Optional[int] = None,
            num_classes: Optional[int] = None,
    ):
        assert num_classes is not None, 'num_classes must be set'
        # config default graph storage type
        self.gtype = 'dgl'
        

        if self.gtype == 'dgl':
            if isinstance(g, DGLGraph):
                self.g = g
                self.num_classes = num_classes

            elif isinstance(g, Tensor):
                assert g.shape[0] == 2
                self.g = dgl.graph(
                    g,
                    num_nodes=num_nodes,
                )
                self.num_classes = num_classes

                self.g.ndata['feat'] = torch.rand(
                    (num_nodes, num_features),
                    dtype=torch.float32,
                )
                self.g.ndata['label'] = torch.randint(num_classes, (num_nodes,))

            else:
                self.g = dgl.graph(
                    ([0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [1, 3, 1, 2, 3, 4, 7, 0, 9]),
                    num_nodes=10,
                )
                self.num_classes = 4

                self.g.ndata['feat'] = torch.rand(
                    (10, 128),
                    dtype=torch.float32,
                )
                self.g.ndata['label'] = torch.randint(4, (10,)) 
            self.g = dgl.add_self_loop(self.g)
            self.g = dgl.to_bidirected(self.g)
            self.g.create_formats_()

        else:
            raise NotImplementedError('fix me')
        

# functions for graph operation

    def to(self, d: list[int] | int):
        if isinstance(d, int):
            self.g = self.g.to(d)

        elif isinstance(d, list):
            raise NotImplementedError('fix me')
        
        return self
    

    def out_edges(self, target_nodes: Tensor):
        target_nodes = target_nodes.to(device=self.g.device)
        u, v = self.g.out_edges(target_nodes)
        return (u, v)
    

    def out_nodes(self, target_nodes: Tensor):
        target_nodes = target_nodes.to(device=self.g.device)
        _, v = self.g.out_edges(target_nodes)
        return v.unique()
    

    def khop_out_nodes(self, target_nodes: Tensor, num_hops: int):
        target_nodes = target_nodes.to(self.g.device)
        aff_map = torch.zeros(self.g.num_nodes(), device=self.g.device, dtype=torch.bool)
        aff_map[target_nodes] = True
        for _ in range(num_hops - 1):
            u, v = self.g.out_edges(target_nodes)
            aff_map[v] = True
        aff_nodes = torch.where(aff_map)[0]
        return aff_nodes.contiguous()


    @property
    def num_edges(self):
        return self.g.num_edges()
    

    @property
    def device(self):
        return self.g.device