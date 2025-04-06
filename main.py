import os
import sys
import logging

import dgl
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

# types
from typing import List, Tuple
from dgl import DGLGraph

# self defined
from lib.convs import GraphConvReduce
from lib.config import Config
from lib.models import model_factory
from lib.app import r_sample, r_partial
from lib.utils import (
    download_dataset, 
    split_randperm, 
    clk_begin, 
    clk_end, 
    free_ram,
    tsstr,
    PERF_LOG,
)


logging.addLevelName(PERF_LOG, 'PERF')

perf_file_handler = logging.FileHandler('logs/NeutronRT.perf.log')
log_file_handler = logging.FileHandler('logs/NeutronRT.log')
cmd_handler = logging.StreamHandler(sys.stderr)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[perf_file_handler, log_file_handler, cmd_handler]
)

perf_file_handler.setLevel(PERF_LOG)
perf_file_handler.setFormatter(
    logging.Formatter('%(message)s')
)

log_file_handler.setLevel(logging.DEBUG)
log_file_handler.setFormatter(
    logging.Formatter('[%(levelname)s@%(asctime)s] %(message)s')
)

cmd_handler.setLevel(logging.DEBUG)
cmd_handler.setFormatter(
    logging.Formatter('%(message)s')
)


def run_base(cfg: Config, g: DGLGraph, model: nn.Module, spawn: bool = False):
    if spawn:
        cfg.fanouts = [5, 5]
        mp.spawn(r_sample, (cfg, g, model))

        cfg.fanouts = [10, 10]
        mp.spawn(r_sample, (cfg, g, model))

        mp.spawn(r_sample, (cfg, g, model, True, True))

        mp.spawn(r_sample, (cfg, g, model, True))
    else:
        cfg.fanouts = [5, 5]
        r_sample(0, cfg, g, model)

        cfg.fanouts = [10, 10]
        r_sample(0, cfg, g, model)
        
        r_sample(0, cfg, g, model, True, True)
        
        r_sample(0, cfg, g, model, True)


def run_part(cfg: Config, g: DGLGraph, model: nn.Module, spawn: bool = False):
    if spawn:
        mp.spawn(r_partial, (cfg, g, model))
    else:
        r_partial(0, cfg, g, model)


if __name__ == '__main__':
    cfg = Config()
    logging.info(f'configuration: {cfg}')

    # thread limiting to avoid resource competition
    omp_num_thread = str(mp.cpu_count() // 2 // len(cfg.gpus))
    os.environ['OMP_NUM_THREADS'] = omp_num_thread
    logging.info(f'OMP_NUM_THREADS: {omp_num_thread}')

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    logging.info('dataloading begin')
    g = download_dataset(cfg.dataset_name, cfg.dataset_root)
    logging.info('dataloading end')
    logging.info(f'graph: {g}')
    logging.info(f'feat: {tsstr(g.ndata["feat"])}')
    logging.info(f'label: {tsstr(g.ndata["label"])}')

    logging.info(f'create random update edges')
    cfg.edges_updated = split_randperm(g.num_edges(), cfg.bs_update, cfg.num_epochs)

    num_features = g.ndata['feat'].shape[1]
    num_classes = g.ndata['label'].unique().shape[0]
    logging.info(f'number of features: {num_features}')
    logging.info(f'number of classes: {num_classes}')

    cfg.num_features = num_features
    cfg.num_classes = num_classes

    if cfg.in_memory:
        cfg.bs_update = g.num_edges() // 1000
        logging.info(f'reset the number of edges updated in one batch to {cfg.bs_update} for in memory dataset')

    model_list: List[Tuple[nn.Module, nn.Module]] = []
    for model_name in cfg.model_names:
        normal_model = model_factory(model_name, cfg.num_layers, num_features, cfg.num_hiddens, num_classes)
        reduce_model = model_factory(model_name, cfg.num_layers, num_features, cfg.num_hiddens, num_classes, conv=GraphConvReduce)
        
        normal_model.eval()
        reduce_model.eval()
        
        model_list.append((normal_model, reduce_model))

    for normal_model, reduce_model in model_list:
        run_part(cfg, g, reduce_model)
        run_base(cfg, g, normal_model)