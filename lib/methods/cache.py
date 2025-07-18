import torch as th
import time

from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock

from .basic import Basic


class Cache(Basic):
    def __init__(self, num_layers):
        super().__init__(num_layers)


    def affect_analysis(self, g, updates, samplers, use_softmax = False):
        src, dst = g.edges()
        for upidx, update in enumerate(updates):
            targets = dst[update.to(g.device)]
            affected_nodes = [None for _ in range(self.num_layers)]
            for layer_idx in range(self.num_layers):
                targets = targets.unique()
                affected_nodes[layer_idx] = targets

                if layer_idx != self.num_layers - 1:
                    _, targets = g.out_edges(targets)

            self.collect_nodes.append(affected_nodes)
    

    def layerwise_inference(self, g, ddp, samplers, feats):
        compt_time = 0.0
        feats, _ = feats

        for upidx, nodes in enumerate(self.collect_nodes):
            for layer_idx, targets in enumerate(nodes):

                nloader = DataLoader(
                    g, targets, samplers[layer_idx],
                    device=g.device,
                    batch_size=8192,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                )

                for it, (innodes, outnodes, blocks) in enumerate(nloader):
                    assert len(blocks) == 1
                    block: DGLBlock = blocks[0]

                    h = feats[layer_idx][innodes]
                    h = h.to(ddp.device)
                    block = block.to(ddp.device)

                    ddp_begin = time.perf_counter()
                    h = ddp(layer_idx, block, h)
                    compt_time += time.perf_counter() - ddp_begin

                    if layer_idx != ddp.module.num_layers - 1:
                        feats[layer_idx][outnodes].copy_(h, non_blocking=True)
                    else:
                        h = h.to('cpu')

        return compt_time