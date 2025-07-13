import torch as th
import dgl
import time

from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock

from .basic import Basic


class Incremental(Basic):
    def __init__(self, num_layers):
        super().__init__(num_layers)


    def affect_analysis(self, g, update, samplers):
        affected_nodes_map = th.zeros(g.num_nodes(), dtype=th.bool, device=g.device)
        affected_nodes = [None for _ in range(self.num_layers)]
        affected_subgraphs = [None for _ in range(self.num_layers)]

        src, dst = g.edges()
        eids = update.to(g.device)
        sg = dgl.graph((src[eids], dst[eids]), num_nodes=g.num_nodes(), device=g.device)
        
        for layer_idx in range(self.num_layers):
            _, targets = sg.edges()
            targets = targets.unique()
            affected_nodes_map[targets] = True
            affected_nodes[layer_idx] = affected_nodes_map.clone()

            affected_subgraphs[layer_idx] = sg.clone()
            affected_subgraphs[layer_idx].create_formats_()

            if layer_idx != self.num_layers - 1:
                src, dst = g.out_edges(targets)
                sg.add_edges(src, dst)
    
        self.collect_nodes.append(affected_nodes)
        self.collect_subgraphs.append(affected_subgraphs)


    def layerwise_inference(self, g, ddp, samplers, feats):
        compt_time = 0.0
        feats, aggrs = feats

        for upidx, subgraphs in enumerate(self.collect_subgraphs):
            for layer_idx, sg in enumerate(subgraphs):
                targets = self.collect_nodes[upidx][layer_idx]

                nloader = DataLoader(
                    sg, targets, samplers[layer_idx],
                    device=g.device,
                    batch_size=1024 * 1024 * 8,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                )

                for it, (innodes, outnodes, blocks) in enumerate(nloader):
                    assert len(blocks) == 1
                    block: DGLBlock = blocks[0]

                    h = feats[layer_idx][innodes]
                    block.srcdata['aggr_cache'] = aggrs[layer_idx][innodes]

                    print(aggrs[layer_idx][innodes])
                    print(block.srcdata['aggr_cache'])

                    h = h.to(ddp.device)
                    block = block.to(ddp.device)

                    ddp_begin = time.perf_counter()
                    h = ddp(layer_idx, block, h)
                    compt_time += time.perf_counter() - ddp_begin

                    if isinstance(h, th.Tensor):
                        if layer_idx != ddp.module.num_layers - 1:
                            feats[layer_idx + 1][outnodes].copy_(h, non_blocking=True)
                        else:
                            h = h.to('cpu')
                    else:
                        h, a = h
                        aggrs[layer_idx][outnodes].copy_(a, non_blocking=True)
                        if layer_idx != ddp.module.num_layers - 1:
                            feats[layer_idx + 1][outnodes].copy_(h, non_blocking=True)
                        else:
                            h = h.to('cpu')

        return compt_time

