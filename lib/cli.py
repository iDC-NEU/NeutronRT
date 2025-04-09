import argparse, torch, pathlib


class CommandLineArgument():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--gpu', type=int, default=1,
            help='choose run model on which cuda:<gpu> device',
        )
        parser.add_argument(
            '--sizes', type=str, default='0,0',
            help='set the size of model layers, now only set the number of model layers',
        )
        parser.add_argument(
            '--root', type=str, default='/home/hdd/dataset',
            help='set the root folder of dataset',
        )
        parser.add_argument(
            '--mem-limit', type=int, default=500 * (1024 ** 3),
            help='set the maximum of address space'
        )
        parser.add_argument(
            '--bs-init', type=int, default=1024 * 64,
            help='batch size for cache initialization at a time'
        )
        parser.add_argument(
            '--bs-update', type=int, default=10_000,
            help='batch size for graph update at a time'
        )
        parser.add_argument(
            '--bs-compute', type=int, default=1_000_000,
            help='batch size for update computation at a time'
        )
        parser.add_argument(
            '--num-epochs', type=int, default=10,
            help='times for training and update',
        )
        parser.add_argument(
            '--num-hiddens', type=int, default=128,
            help='dim of model hidden layers'
        )
        parser.add_argument(
            '--num-workers', type=int, default=4,
            help='workers for loaders'
        )
        parser.add_argument(
            '--embd-host-ratio', type=str, default='1,1',
            help='set cache ratio for layers, <= 1 is the ratio but > 1 is the number of nodes cached'
        )
        parser.add_argument('--embd-cuda-ratio', type=str, default='100,100',
            help='set cache ratio for layers, <= 1 is the ratio but > 1 is the number of nodes cached'
        )
        parser.add_argument('--aggr-host-ratio', type=str, default='1,1',
            help='set cache ratio for layers, <= 1 is the ratio but > 1 is the number of nodes cached'
        )
        parser.add_argument('--aggr-cuda-ratio', type=str, default='100,100',
            help='set cache ratio for layers, <= 1 is the ratio but > 1 is the number of nodes cached'
        )
        parser.add_argument('--sample', type=str, default='-1,-1',
            help='the num of neighbours in sample'
        )

        parser.add_argument('--aggr-method', type=str, default='mean',
                            choices=['mean', 'sum', 'amin', 'amax'])
        parser.add_argument('--model', type=str, default="sage",
                            choices=['sage', 'gcn', 'gin', 'gat'])
        parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                            choices=[
                                # small
                                'ogbn-arxiv', 'reddit', 'ogbn-products',
                                
                                # large
                                'twitter', 'ogbn-papers100M', 'com-friendster'
                            ])
        parser.add_argument('--task', type=str, default='none', 
                            choices=['node', 'none'])

        args = parser.parse_args()

        self.device = torch.device(f'cuda:{args.gpu}')
        self.sizes = [int(sz) for sz in args.sizes.split(',')]
        self.num_model_layers = len(self.sizes)
        self.root = pathlib.Path(args.root)
        self.embd_host_ratio = [float(r) for r in args.embd_host_ratio.split(',')]
        self.embd_cuda_ratio = [float(r) for r in args.embd_cuda_ratio.split(',')]
        self.aggr_host_ratio = [float(r) for r in args.aggr_host_ratio.split(',')]
        self.aggr_cuda_ratio = [float(r) for r in args.aggr_cuda_ratio.split(',')]
        self.sample = [int(r) for r in args.sample.split(',')]
        self.num_epochs: int = args.num_epochs
        self.num_hiddens: int = args.num_hiddens
        self.num_workers: int = args.num_workers
        self.bs_init: int = args.bs_init
        self.bs_update: int = args.bs_update
        self.bs_compute: int = args.bs_compute
        self.mem_limit: int = args.mem_limit
        self.aggr_method: str = args.aggr_method
        self.model: str = args.model
        self.dataset: str = args.dataset
        self.task: str = args.task


    def __str__(self):
        infomation = ''
        for k in self.__dict__.keys():
            v = self.__getattribute__(k)
            infomation = f'{k}: {v}\n\t{infomation}'
        return infomation.strip()