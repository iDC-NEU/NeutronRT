import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np

# types
from typing import Type, List, Callable, Optional
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler

# self defined
from .config import Config
from .cache import Cache
from .convs import (
    GraphConvReduce,
    GINConvReduce,
    SAGEConvReduce,
    AGNNConvReduce,
    GMMConvReduce,
    GATConvReduce,
)
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
    'partial-gin': GINConvReduce,
    'partial-sage': SAGEConvReduce,
    'partial-agnn': AGNNConvReduce,
    'partial-monet': GMMConvReduce,
    'partial-gat': GATConvReduce,

    'gcn'  : dglnn.GraphConv,
    'gin'  : dglnn.GINConv,
    'sage' : dglnn.SAGEConv,
    'agnn' : dglnn.AGNNConv,
    'monet': dglnn.GMMConv,
    'gat'  : dglnn.GATConv,
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
    model_type = model_name.split('-')[-1]
    conv = kwargs.get('conv', None)
    if conv is None:
        conv = conv_map.get(model_name, None)
        if conv is None:
            logging.error(f'{model_name} is not supported currently')
            return None

    if model_type == 'gcn':
        model = GCN(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
        )
    elif model_type == 'gin':
        model = GIN(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
        )
    elif model_type == 'sage':
        model = SAGE(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
            aggregator_type=kwargs.get('aggregator_type', 'mean')
        )
    elif model_type == 'agnn':
        model = AGNN(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
        )
    elif model_type == 'monet':
        model = MONET(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
        )
    elif model_type == 'gat':
        model = GAT(
            model_name=model_name,
            num_layers=num_layers,
            num_features=num_features,
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            conv=conv,
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
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')

        self.layers = nn.ModuleList()
        self.layers.append(conv(num_features, num_hiddens, activation=F.relu))
        for _ in range(num_layers - 2):
            self.layers.append(conv(num_hiddens, num_hiddens, activation=F.relu))
        self.layers.append(conv(num_hiddens, num_classes))

        self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
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
                for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h)
                    si.clk_end('compute')
                    h_list[it] = h
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
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
    def partial_infer(self, cfg, g, caches):
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
                        cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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


class MLP(nn.Module):
    def __init__(
            self,
            num_layers: int,
            in_feats: int,
            hid_feats: int,
            out_feats: int,
            bias: bool = False,
            activation: Optional[Callable] = None,
    ):
        super().__init__()
        super().__setattr__('num_layers', num_layers)
        super().__setattr__('in_feats', in_feats)
        super().__setattr__('hid_feats', hid_feats)
        super().__setattr__('out_feats', out_feats)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hid_feats, bias=bias))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hid_feats, hid_feats, bias=bias))
        self.layers.append(nn.Linear(hid_feats, out_feats, bias=bias))
        self.norm = nn.BatchNorm1d(hid_feats)
        self.activation = activation


    def forward(self, x: torch.Tensor):
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h)
            if layer_idx != self.num_layers - 1:
                h = self.norm(h)
                if self.activation is not None:
                    h = self.activation(h)
        return h


class GIN(BasicModel):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[dglnn.GINConv],
            dropout: float = 0.5,
            num_mlp_layers: int = 2,
    ):
        super().__init__(model_name, num_layers)
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'{num_mlp_layers} mlp layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(conv(
            MLP(num_mlp_layers, num_features, num_hiddens, num_hiddens, activation=F.relu),
            learn_eps=False,
        ))
        self.norms.append(nn.BatchNorm1d(num_hiddens))
        for _ in range(num_layers - 2):
            self.layers.append(conv(
                MLP(num_layers, num_hiddens, num_hiddens, num_hiddens, activation=F.relu),
                learn_eps=False,
            ))
            self.norms.append(nn.BatchNorm1d(num_hiddens))
        self.layers.append(conv(
            MLP(num_layers, num_hiddens, num_hiddens, num_classes, activation=F.relu),
            learn_eps=False,
        ))
        self.norms.append(nn.BatchNorm1d(num_classes))

        self.dropout = nn.Dropout(p=dropout)


    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
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

            if nloader.device == torch.device('cpu'):
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
                for layer_idx, (layer, norm, block) in enumerate(zip(self.layers, self.norms, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    si.clk_begin('compute')
                    h = layer(block, h)
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
                    si.clk_end('compute')
                    h_list[it] = h
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = unique_targets.to(g.device)

            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
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
                    h = layer(block, h)
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
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
    def partial_infer(self, cfg, g, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        sampler = MultiLayerFullNeighborSampler(1)
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            src, dst = g.edges()
            affect_edges = update_batch.to(g.device)
            subg = dgl.graph((src[affect_edges], dst[affect_edges]), num_nodes=g.num_nodes(), device=g.device)

            for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
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
                    if cfg.aggr_method == 'mean':
                        block.dstdata['old_deg'] = g.in_degrees(output_nodes)
                        # TODO: use accurate deg
                        block.dstdata['new_deg'] = block.in_degrees(output_nodes) + block.dstdata['old_deg']

                    si.clk_begin('compute')
                    h = layer(block, h)
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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
        

class SAGE(BasicModel):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[dglnn.SAGEConv],
            aggregator_type: str,
            dropout: float = 0.5,
    ):
        super().__init__(model_name, num_layers)
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')

        self.layers = nn.ModuleList()
        self.layers.append(conv(num_features, num_hiddens, aggregator_type, activation=F.relu))
        for _ in range(num_layers - 2):
            self.layers.append(conv(num_hiddens, num_hiddens, aggregator_type,  activation=F.relu))
        self.layers.append(conv(num_hiddens, num_classes, aggregator_type))

        self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
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
                for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h)
                    h_list[it] = h
                    si.clk_end('compute')
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
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
    def partial_infer(self, cfg, g, caches):
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
                        cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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


class AGNN(BasicModel):
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
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')

        self.layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.layers.append(conv())
        self.linears.append(nn.Linear(num_features, num_hiddens))
        for _ in range(num_layers - 2):
            self.layers.append(conv())
            self.linears.append(nn.Linear(num_hiddens, num_hiddens))
        self.layers.append(conv())
        self.linears.append(nn.Linear(num_hiddens, num_classes))

        # self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
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
                for layer_idx, (layer, linear, block) in enumerate(zip(self.layers, self.linears, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    si.clk_begin('compute')
                    h = layer(block, h)
                    h = linear(h)
                    if layer_idx != self.num_layers - 1:
                        h = F.relu(h)
                    si.clk_end('compute')
                    h_list[it] = h
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = unique_targets.to(g.device)

            for layer_idx, (layer, linear) in enumerate(zip(self.layers, self.linears)):
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
                    h = layer(block, h)
                    h = linear(h)
                    if layer_idx != self.num_layers - 1:
                        h = F.relu(h)
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
    def partial_infer(self, cfg, g, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        sampler = MultiLayerFullNeighborSampler(1)
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            src, dst = g.edges()
            affect_edges = update_batch.to(g.device)
            subg = dgl.graph((src[affect_edges], dst[affect_edges]), num_nodes=g.num_nodes(), device=g.device)

            for layer_idx, (layer, linear) in enumerate(zip(self.layers, self.linears)):
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
                    h = layer(block, h)
                    h = linear(h)
                    if layer_idx != self.num_layers - 1:
                        h = F.relu(h)
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        # cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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


class MONET(BasicModel):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[dglnn.GraphConv],
            dim: int = 1,
            n_kernels: int = 1,
            dropout: float = 0.5,
    ):
        super().__init__(model_name, num_layers)
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')

        self.layers = nn.ModuleList()
        self.pseudos = nn.ModuleList()
        self.layers.append(conv(num_features, num_hiddens, dim, n_kernels))
        self.pseudos.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        for _ in range(num_layers - 2):
            self.layers.append(conv(num_hiddens, num_hiddens, dim, n_kernels))
            self.pseudos.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        self.layers.append(conv(num_hiddens, num_classes, dim, n_kernels))
        self.pseudos.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))

        self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        us, vs = g.edges(order="eid")
        udeg = 1 / torch.sqrt(g.in_degrees(us).float())
        vdeg = 1 / torch.sqrt(g.in_degrees(vs).float())
        e_pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)

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
                for layer_idx, (layer, pseudo, block) in enumerate(zip(self.layers, self.pseudos, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    e_psd = e_pseudo[block.edata[dgl.EID]].to('cuda')
                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h, pseudo(e_psd))
                    si.clk_end('compute')
                    h_list[it] = h
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        us, vs = g.edges(order="eid")
        udeg = 1 / torch.sqrt(g.in_degrees(us).float())
        vdeg = 1 / torch.sqrt(g.in_degrees(vs).float())
        e_pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = unique_targets.to(g.device)

            for layer_idx, (layer, pseudo) in enumerate(zip(self.layers, self.pseudos)):
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
                    e_psd = e_pseudo[block.edata[dgl.EID]].to('cuda')

                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h, pseudo(e_psd))
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
    def partial_infer(self, cfg, g, caches):
        si = StatisticInformation()

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        us, vs = g.edges(order="eid")
        udeg = 1 / torch.sqrt(g.in_degrees(us).float())
        vdeg = 1 / torch.sqrt(g.in_degrees(vs).float())
        e_pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)

        sampler = MultiLayerFullNeighborSampler(1)
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')

            src, dst = g.edges()
            affect_edges = update_batch.to(g.device)
            subg = dgl.graph((src[affect_edges], dst[affect_edges]), num_nodes=g.num_nodes(), device=g.device)

            for layer_idx, (layer, pseudo) in enumerate(zip(self.layers, self.pseudos)):
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
                    e_psd = e_pseudo[block.edata[dgl.EID]].to('cuda')

                    si.clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h, pseudo(e_psd))
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        # cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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


# only support mean on multi-heads
class GAT(BasicModel):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[dglnn.GraphConv],
            num_heads: int = 1,
    ):
        super().__init__(model_name, num_layers)
        super().__setattr__('num_features', num_features)
        super().__setattr__('num_hiddens', num_hiddens)
        super().__setattr__('num_classes', num_classes)

        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')

        self.layers = nn.ModuleList()
        self.layers.append(conv(num_features, num_hiddens, num_heads, activation=F.elu))
        for _ in range(num_layers - 2):
            self.layers.append(conv(num_hiddens, num_hiddens, num_heads, activation=F.elu))
        self.layers.append(conv(num_hiddens, num_classes, num_heads))

        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg, g, sampler):
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
                for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    si.clk_begin('compute')
                    h = layer(block, h)
                    h = h.mean(1)
                    si.clk_end('compute')
                    h_list[it] = h
                si.clk_end('total')

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
    def sample_cached_infer(self, cfg, g, samplers, caches):
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
                    h = layer(block, h)
                    h = h.mean(1)
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
    def partial_infer(self, cfg, g, caches):
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
                    h = layer(block, h)
                    h = h.mean(1)
                    si.clk_end('compute')

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        # cur_a_cache.cache[output_nodes].copy_(block.dstdata['neigh'], non_blocking=True)

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