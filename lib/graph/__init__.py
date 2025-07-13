import logging
import torch as th

from typing import List
from torch import Tensor


def random_eidperm(num_edges: int, bs_update: int, num_snapshots: int) -> List[Tensor]:
    if bs_update * num_snapshots > num_edges:
        bs_update = num_edges // num_snapshots
        logging.info(f'reset the number of edges updated in one batch to {bs_update} because the total update edges exceed the graphs limit')

    rdperm = th.randperm(num_edges)
    rdperm = rdperm[:bs_update * num_snapshots]
    rdperm = rdperm.reshape((num_snapshots, bs_update))
    rdperm = [perm for perm in rdperm]
    return rdperm