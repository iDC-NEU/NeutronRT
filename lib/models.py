import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np

# types
from typing import Type, List
from abc import ABC
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler

# self defined
from .config import Config
from .cache import Cache
from .convs import GraphConvReduce, GATv2ConvReduce
from .utils import (
    StatisticInformation,
    tsstr,
    final_affect_nodes,
    next_affect_edges,
    next_affect_nodes,
    PERF_LOG
)

conv_map = {
    'partial-gcn': GraphConvReduce,
    'partial-gat': GATv2ConvReduce,

    'gcn'  : dglnn.GraphConv,
    'gin'  : dglnn.GINConv,
    'sage' : dglnn.SAGEConv,
    'agnn' : dglnn.AGNNConv,
    'monet': dglnn.GMMConv,

    'gat'  : dglnn.GATv2Conv,
}

class BasicModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
    ):
        super().__init__()
        super().__setattr__('model_name', model_name)
        super().__setattr__('num_layers', num_layers)

    @torch.no_grad()
    def sample_infer(self, cfg: Config, g: DGLGraph, sampler: NeighborSampler): ...

    @torch.no_grad()
    def sample_cached_infer(self, cfg: Config, g: DGLGraph, samplers: List[NeighborSampler], caches: Cache): ...

    @torch.no_grad()
    def partial_infer(self, cfg: Config, g: DGLGraph, caches: Cache): ...


def model_factory(
        model_name: str,
        num_layers: int,
        num_features: int,
        num_hiddens: int,
        num_classes: int,
        **kwargs
) -> "BasicModel":
    model_name_split = model_name.split('-')
    if model_name_split[0] == 'partial':
        if model_name_split[-1] == 'gcn':
            model = GCN(
                model_name=model_name,
                num_layers=num_layers,
                num_features=num_features,
                num_hiddens=num_hiddens,
                num_classes=num_classes,
                conv=GraphConvReduce,
                dropout=kwargs.get('dropout', 0.5)
            )
        else:
            logging.error(f'{model_name} is not supported currently')
            return None
    else:
        if model_name_split[-1] == 'gcn':
            model = GCN(
                model_name=model_name,
                num_layers=num_layers,
                num_features=num_features,
                num_hiddens=num_hiddens,
                num_classes=num_classes,
                conv=dglnn.GraphConv,
                dropout=kwargs.get('dropout', 0.5)
            )
        else:
            logging.error(f'{model_name} is not supported currently')
            return None
    return model


class GCN(BasicModel):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[dglnn.GraphConv],
            dropout: float = 0.5,
    ):
        super().__init__(model_name, num_layers)
        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')
        
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        self.layers.append(conv(num_features, num_hiddens, activation=F.relu))
        for _ in range(num_layers - 2):
            self.layers.append(conv(num_hiddens, num_hiddens, activation=F.relu))
        self.layers.append(conv(num_hiddens, num_classes))

        self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg: Config, g: DGLGraph, sampler: NeighborSampler):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')
            
            # affect analysis
            logging.debug(f'affect analysis begin')
            si.clk_begin('affect')
            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = final_affect_nodes(g, unique_targets, self.num_layers)
            nodes_involved_bitmap[affect_nodes] = True
            si.clk_end('affect')
            logging.debug(f'affect analysis end')
            logging.debug(f'affect nodes: {tsstr(affect_nodes)}')
            si.count('nodes_affected', affect_nodes.shape[0])
            
            # sample
            nloader = DataLoader(
                g,
                affect_nodes,
                sampler,
                device='cuda' if cfg.in_memory else 'cpu',
                batch_size=cfg.bs_compute,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=cfg.use_uva,
            )

            if cfg.numa is not None and nloader.device == torch.device('cpu'):
                nloader.enable_cpu_affinity(
                    loader_cores=cfg.numa[:24],
                    compute_cores=cfg.numa[24:],
                )

            # computation
            logging.debug(f'computation begin')
            h_list = []
            si.clk_begin('total')
            for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                h_list.append(blocks[0].srcdata['feat'].to('cuda'))
                si.clk_begin('compute')
                for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h)
                    h_list[it] = h
                si.clk_end(['compute', 'total'])

                for b in blocks:
                    si.count('nodes_sampled', b.num_src_nodes())
                    si.count('edges_sampled', b.num_edges())
                    edges_involved_bitmap[b.edges('eid')] = True
                nodes_involved_bitmap[input_nodes] = True

                si.clk_begin('total')
            logging.debug(f'computation end')

            si.count('nodes_involved_unique', nodes_involved_bitmap.sum())
            si.count('edges_involved_unique', edges_involved_bitmap.sum())
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()
            si.clk_end_round()
        logging.debug('inference end')
        si.summary(rounds=cfg.num_snapshots)


    @torch.no_grad()
    def sample_cached_infer(self, cfg: Config, g: DGLGraph, samplers: List[NeighborSampler], caches: Cache):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = unique_targets.to(g.device)

            for layer_idx, layer in enumerate(self.layers):
                logging.debug(f'affect nodes: {tsstr(affect_nodes)}')
                si.count('nodes_affected', affect_nodes.shape[0])

                # sample
                nloader = DataLoader(
                    g,
                    affect_nodes,
                    samplers[layer_idx],
                    device='cuda',
                    batch_size=cfg.bs_compute,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                    use_uva=cfg.use_uva,
                )

                cur_h_cache = caches.h_cache[layer_idx]

                # computation
                logging.debug(f'computation begin')
                si.clk_begin('total')
                for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                    h = cur_h_cache.fetch(input_nodes, 'cuda')

                    if not h.is_cuda:
                        h = h.to('cuda')

                    # only sample 1 hop
                    assert len(blocks) == 1, 'multi-hop sample'
                    block = blocks[0]

                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h)
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)

                    si.clk_end('total')

                    for b in blocks:
                        si.count('nodes_sampled', b.num_src_nodes())
                        si.count('edges_sampled', b.num_edges())
                        edges_involved_bitmap[b.edges('eid')] = True
                    nodes_involved_bitmap[input_nodes] = True
                    si.clk_begin('total')
                logging.debug(f'computation end')

                # affect analysis
                if layer_idx != self.num_layers - 1:
                    logging.debug(f'affect analysis begin')
                    si.clk_begin('affect')
                    nxt_aff_nodes = next_affect_nodes(g, affect_nodes)
                    nxt_aff_nodes = nxt_aff_nodes.to(affect_nodes.device)
                    affect_nodes = torch.cat((nxt_aff_nodes, affect_nodes)).unique()
                    si.clk_end('affect')
                    logging.debug(f'affect analysis end')

            si.count('nodes_involved_unique', nodes_involved_bitmap.sum())
            si.count('edges_involved_unique', edges_involved_bitmap.sum())
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()
            si.clk_end_round()
        logging.debug('inference end')
        si.summary(rounds=cfg.num_snapshots)


    @torch.no_grad()
    def partial_infer(self, cfg: Config, g: DGLGraph, caches: Cache):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        sampler = MultiLayerFullNeighborSampler(1)
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            src, dst = g.edges()
            affect_edges = update_batch.to(g.device)
            subg = dgl.graph((src[affect_edges], dst[affect_edges]), num_nodes=g.num_nodes(), device=g.device)

            for layer_idx, layer in enumerate(self.layers):
                logging.debug(f'affect edges #{layer_idx}: {tsstr(affect_edges)}')

                cur_h_cache = caches.h_cache[layer_idx]
                cur_a_cache = caches.a_cache[layer_idx]

                src, dst = subg.edges()
                affect_nodes = dst.unique()
                logging.debug(f'affect nodes: {tsstr(affect_nodes.size(0))}')
                si.count('nodes_affected', affect_nodes.shape[0])

                # sample
                nloader = DataLoader(
                    subg,
                    affect_nodes,
                    sampler,
                    device='cuda',
                    batch_size=cfg.bs_compute,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    use_ddp=True,
                    use_uva=cfg.use_uva,
                )

                # computation
                logging.debug(f'computation begin')
                si.clk_begin('total')
                for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                    h = cur_h_cache.fetch(input_nodes, 'cuda')

                    # only sample 1 hop
                    assert len(blocks) == 1, 'multi-hop sample'
                    block = blocks[0]
                    block.dstdata['aggr_cache'] = cur_a_cache.fetch(output_nodes, 'cuda')

                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h)
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        cur_a_cache.cache[output_nodes].copy_(block.dstdata['aggr_cache'], non_blocking=True)

                    si.clk_end('total')

                    for b in blocks:
                        si.count('nodes_sampled', b.num_src_nodes())
                        si.count('edges_sampled', b.num_edges())
                        edges_involved_bitmap[b.edges('eid')] = True
                    nodes_involved_bitmap[input_nodes] = True
                    si.clk_begin('total')
                logging.debug(f'computation end')

                # affect analysis
                if layer_idx != self.num_layers - 1:
                    logging.debug(f'affect analysis begin')
                    si.clk_begin('affect')
                    (nxt_aff_src, nxt_aff_dst) = next_affect_edges(g, affect_nodes)
                    subg.add_edges(nxt_aff_src, nxt_aff_dst)
                    si.clk_end('affect')
                    logging.debug(f'affect analysis end')

            si.count('nodes_involved_unique', nodes_involved_bitmap.sum())
            si.count('edges_involved_unique', edges_involved_bitmap.sum())
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()
            si.clk_end_round()
        logging.debug('inference end')
        si.summary(rounds=cfg.num_snapshots)