import torch
import torch.distributed as dist

from ..dataset.loader import load_dataset
from ..graph.metagraph import MetaGraph

from dgl.dataloading import NeighborSampler, DataLoader
from typing import Any
from torch import Tensor
from dgl.heterograph import DGLBlock



class MasterProc():
    def __init__(self, config: dict[str, Any]):
        self.method: str = config['method']
        self.root: str = config['root']
        self.dataset: str = config['dataset']

        self.in_memory: bool = config['in_memory']
        self.use_uva: bool = config['use_uva']

        self.world_size: int = config['world_size']

        self.bs_update: int = config['bs_update']
        self.bs_compute: int = config['bs_compute']

        self.model_dims: list[int] = config['model_dims']
        self.num_model_layers: int = len(self.model_dims)
        self.num_snapshots: int = config['num_snapshots']

        self.sampler: list[NeighborSampler] = []
        for layer_idx, fanout in enumerate(config['fanouts']):
            self.sampler.append(NeighborSampler(fanout))



    def __call__(self, rank: int, world_size: int, devices: list[int]):
        g = load_dataset(self.dataset, self.root)

        # now only use one device for storage
        devices = [devices[0]]


        if self.in_memory:
            g.to(devices[0])

        
        print(f'[master #{rank}] graph data load success')

        rdperm = torch.randperm(g.num_edges)
        rdperm = rdperm[:self.bs_update * self.num_snapshots]
        rdperm = rdperm.reshape((self.num_snapshots, self.bs_update))


        for update_edges in rdperm:
            match self.method:
                case 'full':
                    _, dst = update_edges
                    final_affect_nodes = g.khop_out_nodes(dst, self.num_model_layers - 1)
                    self.sample_then_compute(self.num_model_layers - 1)


    def sample_then_compute(self, layer_idx: int, g: MetaGraph, dst: Tensor) -> Tensor:
        sampler = self.sampler[layer_idx]

        if layer_idx == 0:
            pass
        else:
            nloader = DataLoader(
                g,
                dst,
                sampler,
                device=g.device,
                batch_size=self.bs_compute,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=self.use_uva,
            )

            for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                assert len(blocks) == 1, 'layerwise inference process only 1 layer per computation'
                h = self.sample_then_compute(layer_idx - 1, g, input_nodes)

                compute_device = self.slave_select()
                
                block: DGLBlock = blocks[0]
                dist.send_object_list([h, block], dst=compute_device)

                h = torch.zeros(
                    (output_nodes.size(0), self.model_dims[layer_idx]),
                    dtype=h.dtype,
                )
                dist.recv(tensor=h, src=compute_device)
                

                




                    

        
    def slave_select(self):
        return 1