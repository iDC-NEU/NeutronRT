import torch

from ..dataset.loader import load_dataset

from typing import Any


class SlaveProc():
    def __init__(self, config: dict[str, Any]):
        self.model = config['model']


    def __call__(self, rank: int, world_size: int, device: int):
        torch.cuda.set_device(device)

        print(f'[{rank}/{world_size-1}] use device #{device}')