import torch as th
import time

from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock

from .basic import Basic


class Sample(Basic):
    def __init__(self, num_layers):
        super().__init__(num_layers)


    def affect_analysis(self, g, update, samplers):
        affected_nodes_map = th.zeros(g.num_nodes(), dtype=th.bool, device=g.device)
        affected_nodes = [None for _ in range(self.num_layers)]

        src, dst = g.edges()
        targets = dst[update.to(g.device)]
        for layer_idx in range(self.num_layers):
            targets = targets.unique()
            if layer_idx != self.num_layers - 1:
                _, targets = g.out_edges(targets)

        affected_nodes_map[targets] = True
        for layer_idx in reversed(range(self.num_layers)):
            affected_nodes[layer_idx] = affected_nodes_map.clone()

            if layer_idx != 0:
                nloader = DataLoader(
                    g, th.where(affected_nodes_map)[0], samplers[layer_idx],
                    device=g.device,
                    batch_size=1024 * 1024 * 8,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                )

                affected_nodes_map.zero_()
                for innodes, outnodes, blocks in nloader:
                    affected_nodes_map[innodes] = True

        self.collect_nodes.append(affected_nodes)



    def layerwise_inference(self, g, ddp, samplers, feats):
        compt_time = 0.0
        feats, _ = feats
        feats = feats[0]
        h_src = None
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

                dims = ddp.module.num_hiddens if layer_idx != ddp.module.num_layers else ddp.module.num_classes
                
                if layer_idx == 0:
                    h_src = feats
                else:
                    h_src = h_store

                h_store = th.zeros(
                    (g.num_nodes(), dims),
                    dtype=feats.dtype,
                    device=g.device,
                    pin_memory=g.is_pinned(),
                )

                for it, (innodes, outnodes, blocks) in enumerate(nloader):
                    assert len(blocks) == 1
                    block: DGLBlock = blocks[0]
                    block = block.to(ddp.device)

                    h = h_src[innodes]
                    h = h.to(ddp.device)

                    ddp_begin = time.perf_counter()
                    h = ddp(layer_idx, block, h)
                    compt_time += time.perf_counter() - ddp_begin

                    if layer_idx != ddp.module.num_layers - 1:
                        h_store[outnodes] = h.to(g.device)
                    else:
                        h = h.to('cpu')

        return compt_time
