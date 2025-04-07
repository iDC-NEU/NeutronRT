import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np

# types
from typing import Type, List
from dgl import DGLGraph
from dgl.dataloading import NeighborSampler, DataLoader, MultiLayerFullNeighborSampler

# self defined
from .config import Config
from .cache import Cache
from .convs import GraphConvReduce
from .utils import (
    clk_begin,
    clk_end,
    tsstr,
    final_affect_nodes,
    next_affect_edges,
    next_affect_nodes,
    PERF_LOG
)

def model_factory(
        model_name: str,
        num_layers: int,
        num_features: int,
        num_hiddens: int,
        num_classes: int,
        **kwargs
):
    model = None
    if kwargs.get('conv', None) is None:
        activation = kwargs.get('activation', F.relu)

        if model_name.startswith('partial-'):
            if model_name == 'partial-gcn':
                kwargs['conv'] = GraphConvReduce

                kwargs['conv_in_args'] = {
                    'in_feats': num_features,
                    'out_feats': num_hiddens,
                    'activation': kwargs.get('activation', F.relu),
                }
                kwargs['conv_hidden_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_hiddens,
                    'activation': kwargs.get('activation', F.relu),
                }
                kwargs['conv_out_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_classes,
                }
            else:
                logging.error(f'{model_name} not support')
                return None
        else:
            if model_name == 'gcn':
                kwargs['conv'] = dglnn.GraphConv

                kwargs['conv_in_args'] = {
                    'in_feats': num_features,
                    'out_feats': num_hiddens,
                    'activation': kwargs.get('activation', F.relu),
                }
                kwargs['conv_hidden_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_hiddens,
                    'activation': kwargs.get('activation', F.relu),
                }
                kwargs['conv_out_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_classes,
                }
            elif model_name == 'gat':
                kwargs['conv'] = dglnn.GATv2Conv
                kwargs['conv_in_args'] = {
                    'in_feats': num_features,
                    'out_feats': num_hiddens,
                    'num_heads': kwargs.get('num_heads', 1),
                    'feat_drop': kwargs.get('feat_drop', 0.6),
                    'attn_drop': kwargs.get('attn_drop', 0.6),
                    'activation': kwargs.get('activation', F.elu),
                }
                kwargs['conv_hidden_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_hiddens,
                    'num_heads': kwargs.get('num_heads', 1),
                    'feat_drop': kwargs.get('feat_drop', 0.6),
                    'attn_drop': kwargs.get('attn_drop', 0.6),
                    'activation': kwargs.get('activation', F.elu),
                }
                kwargs['conv_out_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_classes,
                    'num_heads': kwargs.get('num_heads', 1),
                    'feat_drop': kwargs.get('feat_drop', 0.6),
                    'attn_drop': kwargs.get('attn_drop', 0.6),
                }
            elif model_name == 'gin':
                kwargs['conv'] = dglnn.GINConv
                lin = kwargs.get('lin', nn.Linear)
                aggregator_type = kwargs.get('aggregator_type', 'sum')

                kwargs['conv_in_args'] = {
                    'apply_func': lin(num_features, num_hiddens),
                    'aggregator_type': aggregator_type,
                    'activation': activation,
                }
                kwargs['conv_hidden_args'] = {
                    'apply_func': lin(num_hiddens, num_hiddens),
                    'aggregator_type': aggregator_type,
                    'activation': activation,
                }
                kwargs['conv_out_args'] = {
                    'apply_func': lin(num_hiddens, num_classes),
                    'aggregator_type': aggregator_type,
                }
            elif model_name == 'sage':
                kwargs['conv'] = dglnn.SAGEConv
                aggregator_type = kwargs.get('aggregator_type', 'mean')

                kwargs['conv_in_args'] = {
                    'in_feats': num_features,
                    'out_feats': num_hiddens,
                    'aggregator_type': aggregator_type,
                    'activation': activation,
                }
                kwargs['conv_hidden_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_hiddens,
                    'aggregator_type': aggregator_type,
                    'activation': activation,
                }
                kwargs['conv_out_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_classes,
                    'aggregator_type': aggregator_type,
                }
            elif model_name == 'agnn':
                kwargs['conv'] = dglnn.AGNNConv

                kwargs['conv_in_args'] = {}
                kwargs['conv_hidden_args'] = {}
                kwargs['conv_out_args'] = {}
            elif model_name == 'monet':
                kwargs['conv'] = dglnn.GMMConv
                dim = kwargs.get('dim', 1)
                n_kernels = kwargs.get('n_kernels', 1)
                aggregator_type = kwargs.get('aggregator_type', 'sum')

                kwargs['conv_in_args'] = {
                    'in_feats': num_features,
                    'out_feats': num_hiddens,
                    'dim': dim,
                    'n_kernels': n_kernels,
                    'aggregator_type': aggregator_type,
                }
                kwargs['conv_hidden_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_hiddens,
                    'dim': dim,
                    'n_kernels': n_kernels,
                    'aggregator_type': aggregator_type,
                }
                kwargs['conv_out_args'] = {
                    'in_feats': num_hiddens,
                    'out_feats': num_classes,
                    'dim': dim,
                    'n_kernels': n_kernels,
                    'aggregator_type': aggregator_type,
                }
            else:
                logging.error(f'{model_name} not support')
                return None

    model = Basic(
        model_name=model_name,
        num_layers=num_layers,
        num_features=num_features,
        num_hiddens=num_hiddens,
        num_classes=num_classes,
        **kwargs,
    )
    return model


class Basic(nn.Module):
    def __init__(
            self,
            model_name: str,
            num_layers: int,
            num_features: int,
            num_hiddens: int,
            num_classes: int,
            conv: Type[nn.Module],
            dropout: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        logging.debug(f'model creation begin')
        logging.debug(f'{num_layers} layers')
        logging.debug(f'dims: ({num_features}, {num_hiddens}, {num_classes})')
        logging.debug(f'dropout: {dropout}')
        
        self.model_name = model_name

        self.num_layers = num_layers
        self.num_features = num_features
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        self.layers.append(conv(**kwargs['conv_in_args']))
        for _ in range(num_layers - 2):
            self.layers.append(conv(**kwargs['conv_hidden_args']))
        self.layers.append(conv(**kwargs['conv_out_args']))

        if self.model_name == 'monet':
            self.pseudo_proj = kwargs.get('pseudo_proj', nn.ModuleList())
            if len(self.pseudo_proj) == 0:
                self.pseudo_proj.append(nn.Sequential(nn.Linear(2, kwargs['conv_in_args']['dim']), nn.Tanh()))
                for _ in range(num_layers - 2):
                    self.pseudo_proj.append(nn.Sequential(nn.Linear(2, kwargs['conv_in_args']['dim']), nn.Tanh()))
                self.pseudo_proj.append(nn.Sequential(nn.Linear(2, kwargs['conv_in_args']['dim']), nn.Tanh()))

        self.dropout = nn.Dropout(p=dropout)
        logging.debug(f'model creation end')
    

    @torch.no_grad()
    def sample_infer(self, cfg: Config, g: DGLGraph, sampler: NeighborSampler):
        nodes_affected = 0
        nodes_sampled = 0
        edges_sampled = 0
        nodes_involved_unique = 0
        edges_involved_unique = 0

        affxt_sum_time = []
        compt_sum_time = []
        total_sum_time = []

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        if self.model_name == 'monet':
            udeg = 1 / torch.sqrt(g.in_degrees(src).float())
            vdeg = 1 / torch.sqrt(g.in_degrees(dst).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1).to('cuda')

        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')
            affxt_time = []
            compt_time = []
            total_time = []
            
            # affect analysis
            logging.debug(f'affect analysis begin')
            clk_begin('affect')
            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = final_affect_nodes(g, unique_targets, self.num_layers)
            nodes_involved_bitmap[affect_nodes] = True
            affxt_time.append(clk_end('affect'))
            logging.debug(f'affect analysis end')
            logging.debug(f'affect nodes: {tsstr(affect_nodes)}')

            nodes_affected += affect_nodes.shape[0]
            
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
            clk_begin('total')
            for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                h_list.append(blocks[0].srcdata['feat'].to('cuda'))
                clk_begin('compute')
                for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                    block = block.to('cuda')
                    h = h_list[it].to('cuda')
                    if self.model_name == 'monet':
                        epseu = pseudo[block.edges('eid').to('cuda')]
                        additional_arguments = {
                            'pseudo': self.pseudo_proj[layer_idx](epseu)
                        }
                    else:
                        additional_arguments = {}
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h = layer(block, h, **additional_arguments)

                    if self.model_name == 'gat':
                        # use mean for now
                        h = torch.mean(h, dim=1)
                    h_list[it] = h
                compt_time.append(clk_end('compute'))
                total_time.append(clk_end('total'))

                for b in blocks:
                    nodes_sampled += b.num_src_nodes()
                    edges_sampled += b.num_edges()
                    edges_involved_bitmap[b.edges('eid')] = True
                nodes_involved_bitmap[input_nodes] = True

                clk_begin('total')
            logging.debug(f'computation end')

            nodes_involved_unique += nodes_involved_bitmap.sum()
            edges_involved_unique += edges_involved_bitmap.sum()
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()

            affxt_sum_time.append(np.sum(affxt_time))
            compt_sum_time.append(np.sum(compt_time))
            total_sum_time.append(np.sum(total_time))
        logging.debug('inference end')

        affxt_sum_time = np.array(affxt_sum_time)
        compt_sum_time = np.array(compt_sum_time)
        total_sum_time = np.array(total_sum_time)

        logging.log(PERF_LOG, f'Computation     time: {np.median(compt_sum_time)}')
        logging.log(PERF_LOG, f'Affect analysis time: {np.median(affxt_sum_time)}')
        logging.log(PERF_LOG, f'Sample+Transfer time: {np.median(total_sum_time - (compt_sum_time))}')
        logging.log(PERF_LOG, f'Nodes  in total: {nodes_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'Edges  in total: {edges_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'aNodes in total: {nodes_affected // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uNodes in total: {nodes_involved_unique // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uEdges in total: {edges_involved_unique // cfg.num_snapshots}')


    @torch.no_grad()
    def sample_cached_infer(self, cfg: Config, g: DGLGraph, samplers: List[NeighborSampler], caches: Cache):
        nodes_affected = 0
        nodes_sampled = 0
        edges_sampled = 0
        nodes_involved_unique = 0
        edges_involved_unique = 0

        affxt_sum_time = []
        compt_sum_time = []
        total_sum_time = []

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        if self.model_name == 'monet':
            udeg = 1 / torch.sqrt(g.in_degrees(src).float())
            vdeg = 1 / torch.sqrt(g.in_degrees(dst).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1).to('cuda')

        src, dst = g.edges()
        logging.debug('inference begin')
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')
            affxt_time = []
            compt_time = []
            total_time = []

            unique_targets = dst[update_batch.to(g.device)].unique()
            affect_nodes = unique_targets.to(g.device)
            
            for layer_idx, layer in enumerate(self.layers):
                logging.debug(f'affect nodes: {tsstr(affect_nodes)}')

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
                clk_begin('total')
                for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                    h = cur_h_cache.fetch(input_nodes, 'cuda')

                    if not h.is_cuda:
                        h = h.to('cuda')

                    # only sample 1 hop
                    assert len(blocks) == 1, 'multi-hop sample'
                    block = blocks[0]

                    if self.model_name == 'monet':
                        epseu = pseudo[block.edges('eid').to('cuda')]
                        additional_arguments = {
                            'pseudo': self.pseudo_proj[layer_idx](epseu)
                        }
                    else:
                        additional_arguments = {}

                    clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)

                    h = layer(block, h, **additional_arguments)

                    if self.model_name == 'gat':
                        # use mean for now
                        h = torch.mean(h, dim=1)
                    compt_time.append(clk_end('compute'))

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)

                    total_time.append(clk_end('total'))

                    for b in blocks:
                        nodes_sampled += b.num_src_nodes()
                        edges_sampled += b.num_edges()
                        edges_involved_bitmap[b.edges('eid')] = True
                    nodes_involved_bitmap[input_nodes] = True

                    clk_begin('total')
                logging.debug(f'computation end')

                # affect analysis
                if layer_idx != self.num_layers - 1:
                    logging.debug(f'affect analysis begin')
                    clk_begin('affect')
                    nxt_aff_nodes = next_affect_nodes(g, affect_nodes)
                    nxt_aff_nodes = nxt_aff_nodes.to(affect_nodes.device)
                    affect_nodes = torch.cat((nxt_aff_nodes, affect_nodes)).unique()
                    affxt_time.append(clk_end('affect'))
                    logging.debug(f'affect analysis end')

            nodes_affected += affect_nodes.shape[0]
            nodes_involved_unique += nodes_involved_bitmap.sum()
            edges_involved_unique += edges_involved_bitmap.sum()
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()

            affxt_sum_time.append(np.sum(affxt_time))
            compt_sum_time.append(np.sum(compt_time))
            total_sum_time.append(np.sum(total_time))
        logging.debug('inference end')

        affxt_sum_time = np.array(affxt_sum_time)
        compt_sum_time = np.array(compt_sum_time)
        total_sum_time = np.array(total_sum_time)

        logging.log(PERF_LOG, f'Computation     time: {np.median(compt_sum_time)}')
        logging.log(PERF_LOG, f'Affect analysis time: {np.median(affxt_sum_time)}')
        logging.log(PERF_LOG, f'Sample+Transfer time: {np.median(total_sum_time - (compt_sum_time))}')
        logging.log(PERF_LOG, f'Nodes  in total: {nodes_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'Edges  in total: {edges_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'aNodes in total: {nodes_affected // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uNodes in total: {nodes_involved_unique // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uEdges in total: {edges_involved_unique // cfg.num_snapshots}')

        
    @torch.no_grad()
    def partial_infer(self, cfg: Config, g: DGLGraph, caches: Cache):
        nodes_affected = 0
        nodes_sampled = 0
        edges_sampled = 0
        nodes_involved_unique = 0
        edges_involved_unique = 0

        affxt_sum_time = []
        compt_sum_time = []
        total_sum_time = []

        nodes_involved_bitmap = torch.zeros(g.num_nodes(), dtype=torch.bool)
        edges_involved_bitmap = torch.zeros(g.num_edges(), dtype=torch.bool)

        src, dst = g.edges()
        if self.model_name == 'monet':
            udeg = 1 / torch.sqrt(g.in_degrees(src).float())
            vdeg = 1 / torch.sqrt(g.in_degrees(dst).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1).to('cuda')

        sampler = MultiLayerFullNeighborSampler(1)
        for update_batch in cfg.edges_updated:
            logging.debug(f'update edges: {tsstr(update_batch)}')
            affxt_time = []
            compt_time = []
            total_time = []

            src, dst = g.edges()
            affect_edges = update_batch.to(g.device)
            subg = dgl.graph((src[affect_edges], dst[affect_edges]), num_nodes=g.num_nodes(), device=g.device)
            if g.is_pinned():
                subg.pin_memory_()

            for layer_idx, layer in enumerate(self.layers):
                logging.debug(f'affect edges #{layer_idx}: {tsstr(affect_edges)}')

                cur_h_cache = caches.h_cache[layer_idx]
                cur_a_cache = caches.a_cache[layer_idx]

                src, dst = subg.edges()
                affect_nodes = dst.unique()
                logging.debug(f'affect nodes: {tsstr(affect_nodes.size(0))}')

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
                clk_begin('total')
                for it, (input_nodes, output_nodes, blocks) in enumerate(nloader):
                    h = cur_h_cache.fetch(input_nodes, 'cuda')
                    a = cur_a_cache.fetch(output_nodes, 'cuda')

                    # only sample 1 hop
                    assert len(blocks) == 1, 'multi-hop sample'
                    block = blocks[0]

                    clk_begin('compute')
                    if layer_idx != self.num_layers - 1:
                        h = self.dropout(h)
                    h, a = layer(block, h, a)

                    if self.model_name == 'gat':
                        # use mean for now
                        h = torch.mean(h, dim=1)
                    compt_time.append(clk_end('compute'))

                    if layer_idx != self.num_layers - 1:
                        nxt_h_cache = caches.h_cache[layer_idx + 1]
                        
                        output_nodes = output_nodes.to(nxt_h_cache.cache.device)
                        nxt_h_cache.cache[output_nodes].copy_(h, non_blocking=True)
                        cur_a_cache.cache[output_nodes].copy_(a, non_blocking=True)

                    total_time.append(clk_end('total'))

                    for b in blocks:
                        nodes_sampled += b.num_src_nodes()
                        edges_sampled += b.num_edges()
                        edges_involved_bitmap[b.edges('eid')] = True
                    nodes_involved_bitmap[input_nodes] = True
                    clk_begin('total')
                logging.debug(f'computation end')

                # affect analysis
                if layer_idx != self.num_layers - 1:
                    logging.debug(f'affect analysis begin')
                    clk_begin('affect')
                    (nxt_aff_src, nxt_aff_dst) = next_affect_edges(g, affect_nodes)
                    subg.add_edges(nxt_aff_src, nxt_aff_dst)
                    affxt_time.append(clk_end('affect'))
                    logging.debug(f'affect analysis end')

            nodes_affected += affect_nodes.shape[0]
            nodes_involved_unique += nodes_involved_bitmap.sum()
            edges_involved_unique += edges_involved_bitmap.sum()
            nodes_involved_bitmap.zero_()
            edges_involved_bitmap.zero_()

            affxt_sum_time.append(np.sum(affxt_time))
            compt_sum_time.append(np.sum(compt_time))
            total_sum_time.append(np.sum(total_time))
        logging.debug('inference end')

        affxt_sum_time = np.array(affxt_sum_time)
        compt_sum_time = np.array(compt_sum_time)
        total_sum_time = np.array(total_sum_time)

        logging.log(PERF_LOG, f'Computation     time: {np.median(compt_sum_time)}')
        logging.log(PERF_LOG, f'Affect analysis time: {np.median(affxt_sum_time)}')
        logging.log(PERF_LOG, f'Sample+Transfer time: {np.median(total_sum_time - (compt_sum_time))}')
        logging.log(PERF_LOG, f'Nodes  in total: {nodes_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'Edges  in total: {edges_sampled // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'aNodes in total: {nodes_affected // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uNodes in total: {nodes_involved_unique // cfg.num_snapshots}')
        logging.log(PERF_LOG, f'uEdges in total: {edges_involved_unique // cfg.num_snapshots}')

            

