import torch as th
import dgl
import time

from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock

from .basic import Basic


class Incremental(Basic):
    def __init__(self, num_layers):
        super().__init__(num_layers)

        self.collect_recomputed = []


    def affect_analysis(self, g, updates, samplers, use_softmax = False):
        src, dst = g.edges()
        for upidx, update in enumerate(updates):
            affected_nodes = [None for _ in range(self.num_layers)]
            affected_subgraphs = [None for _ in range(self.num_layers)]
            if use_softmax:
                recomputed_nodes = [None for _ in range(self.num_layers)]
                former_affected_nodes_map = th.zeros(g.num_nodes(), dtype=th.bool, device=g.device)
                affected_nodes_map = th.zeros(g.num_nodes(), dtype=th.bool, device=g.device)

            update = update.to(g.device)
            sg = dgl.graph((src[update], dst[update]), num_nodes=g.num_nodes(), device=g.device)

            for layer_idx in range(self.num_layers):
                _, targets = sg.edges()
                targets = targets.unique()
                affected_nodes[layer_idx] = targets

                if not use_softmax:
                    affected_nodes[layer_idx] = targets
                else:
                    affected_nodes_map.zero_()
                    affected_nodes_map[targets] = True

                    recomputed_map = th.bitwise_and(affected_nodes_map, former_affected_nodes_map)
                    recomputed_nodes[layer_idx] = th.where(recomputed_map)[0]

                    nonrecomputed_map = th.bitwise_and(affected_nodes_map, th.bitwise_not(former_affected_nodes_map))
                    affected_nodes[layer_idx] = th.where(nonrecomputed_map)[0]
                    
                    former_affected_nodes_map[targets] = True

                sg.create_formats_()
                affected_subgraphs[layer_idx] = sg.clone()

                if layer_idx != self.num_layers - 1:
                    asrc, adst = g.out_edges(targets)
                    sg.add_edges(asrc, adst)
        
            self.collect_nodes.append(affected_nodes)
            self.collect_subgraphs.append(affected_subgraphs)
            if use_softmax:
                self.collect_recomputed.append(recomputed_nodes)


    def layerwise_inference(self, g, ddp, samplers, feats):
        compt_time = 0.0
        feats, aggrs = feats

        for upidx, subgraphs in enumerate(self.collect_subgraphs):
            for layer_idx, sg in enumerate(subgraphs):
                targets = self.collect_nodes[upidx][layer_idx]

                nloader = DataLoader(
                    sg, targets, samplers[layer_idx],
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
                    block.srcdata['aggr_cache'] = aggrs[layer_idx][innodes]

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

                if ddp.module.use_softmax:
                    targets = self.collect_recomputed[upidx][layer_idx]

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

