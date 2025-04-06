import argparse
import pathlib
import json

from .utils import GB


class Config():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--root', type=str, default='/home/hdd/dataset',
            help='set the root folder of dataset',
        )
        parser.add_argument(
            '--aggr', type=str, default='sum',
            choices=['sum', 'mean', 'min', 'max']
        )
        parser.add_argument(
            '--models', type=str, default="gcn",
            choices=[
                'gcn', 'gat', 'sage', 'gin',
            ]
        )
        parser.add_argument(
            '--dataset', type=str, default='ogbn-arxiv',
            choices=[
                # small
                'ogbn-arxiv', 'reddit', 'ogbn-products',
                
                # large
                'twitter', 'ogbn-papers100M', 'com-friendster'
            ]
        )
        parser.add_argument(
            '--gpus', type=str, default='0',
            help='choose devices on which the model will run',
        )
        parser.add_argument('--fanouts', type=str, default='-1',
            help='the num of neighbours sampled per layer'
        )
        parser.add_argument(
            '--bs-update', type=int, default=100_000,
            help='batch size for graph update at a time'
        )
        parser.add_argument(
            '--bs-compute', type=int, default=1_000_000,
            help='batch size for update computation at a time'
        )
        parser.add_argument(
            '--num-layers', type=int, default=2,
            help='set the number of layers',
        )
        parser.add_argument(
            '--num-epochs', type=int, default=4,
            help='times for training and update',
        )
        parser.add_argument(
            '--num-hiddens', type=int, default=128,
            help='dim of model hidden layers'
        )
        parser.add_argument(
            '--num-threads', type=int, default=4,
            help='number of the threads will be used'
        )

        parser.add_argument('--embd-host-ratio', type=float, default=0.0)
        parser.add_argument('--embd-cuda-ratio', type=float, default=0.0)
        parser.add_argument('--aggr-host-ratio', type=float, default=0.0)
        parser.add_argument('--aggr-cuda-ratio', type=float, default=0.0)

        args = parser.parse_args()

        self.dataset_root = args.root
        self.dataset_name: str = args.dataset
        self.aggr_method: str = args.aggr
        self.model_names = args.models.split(',')
        self.gpus = [int(dev) for dev in args.gpus.split(',')]
        self.fanouts = [int(fanout) for fanout in args.fanouts.split(',')]

        # batch size
        self.bs_update: int = args.bs_update
        self.bs_compute: int = args.bs_compute

        # numbers
        self.num_layers: int = args.num_layers
        self.num_epochs: int = args.num_epochs
        self.num_hiddens: int = args.num_hiddens
        self.num_threads: int = args.num_threads

        # cache ratio
        self.embd_host_ratio: float = args.embd_host_ratio
        self.embd_cuda_ratio: float = args.embd_cuda_ratio
        self.aggr_host_ratio: float = args.aggr_host_ratio
        self.aggr_cuda_ratio: float = args.aggr_cuda_ratio

        self.in_memory = bool(self.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'reddit'])
        self.map_location = 'cuda' if self.in_memory else 'cpu'
        self.use_uva = False if self.in_memory else True


    def __str__(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=True)