import os
import toml
import time
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# types
from typing import Any
from lib.dist.master import MasterProc
from lib.dist.slave import SlaveProc


def run(rank: int, world_size: int, config: dict[str, Any]):
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    devices = config['devices']
    num_graph_devices = config['num_graph_devices']
    
    if rank == 0:
        master = MasterProc(config)
        master(rank, world_size, devices[0:num_graph_devices])
    else:
        slave = SlaveProc(config)
        slave(rank, world_size, devices[rank + num_graph_devices - 1])

    dist.destroy_process_group()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='config/default.toml',
        help='set the config file of the server',
    )

    args = parser.parse_args()
    config = args.config
    config = toml.load(config)

    print(config)

    for k, v in config['env'].items():
        os.environ[k] = str(v)

    world_size = config['world_size']

    mp.spawn(run, args=(world_size, config), nprocs=world_size, join=True)

    