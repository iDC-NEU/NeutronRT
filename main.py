import os
import sys
import logging
import pathlib

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

# debugging environment varibales
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# add new log level for perf estimation
logging.addLevelName(PERF_LOG, 'PERF')

perf_file_handler = logging.FileHandler('logs/NeutronRT.perf.log')
log_file_handler = logging.FileHandler('logs/NeutronRT.log')
cmd_handler = logging.StreamHandler(sys.stderr)

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

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[perf_file_handler, log_file_handler, cmd_handler]
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
    logging.info(f'ntypes: {g.ntypes}')
    logging.info(f'feat: {tsstr(g.ndata["feat"])}')
    logging.info(f'label: {tsstr(g.ndata["label"])}')

    logging.info(f'create random update edges')
    cached_update_file = pathlib.Path(f'{cfg.dataset_root}/.cache/{g.num_edges()}.{cfg.bs_update}.{cfg.num_snapshots}.update')
    if cached_update_file.is_file():
        logging.info(f'loading pre generated update file from {cached_update_file}')
        cfg.edges_updated = torch.load(cached_update_file)
    else:
        cfg.edges_updated = split_randperm(g.num_edges(), cfg.bs_update, cfg.num_snapshots)
        torch.save(cfg.edges_updated, cached_update_file)

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
        logging.info(f'create model {model_name}')
        logging.log(PERF_LOG, f'> {model_name} @ {cfg.dataset_name}')
        reduce_model = model_factory(f'partial-{model_name}', cfg.num_layers, num_features, cfg.num_hiddens, num_classes)
        if reduce_model is not None:
            reduce_model = reduce_model.eval()
            logging.info(f'partial-{model_name}: {reduce_model}')
            run_part(cfg, g, reduce_model, spawn=True)

        normal_model = model_factory(model_name, cfg.num_layers, num_features, cfg.num_hiddens, num_classes)
        if normal_model is not None:
            normal_model = normal_model.eval()
            logging.info(f'{model_name}: {normal_model}')
            run_base(cfg, g, normal_model, spawn=True)
        logging.log(PERF_LOG, f'< {model_name} @ {cfg.dataset_name}')
        logging.log(PERF_LOG, f'*****')

    