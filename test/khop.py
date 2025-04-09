import inspect

import torch
import torch_geometric

from torch_geometric.utils import k_hop_subgraph

def ptensor(t: torch.Tensor):
    t_name = None
    for n, v in inspect.currentframe().f_back.f_locals.items():
        if v is t:
            t_name = n
    
    if t_name is None:
        raise ValueError

    print(f'{t_name}: {t.dtype}@{t.device} {t.shape} {t}')

dst = torch.LongTensor([2])
edge_index = torch.LongTensor([[2, 3, 4, 5], 
                               [1, 2, 2, 6]])

subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=dst,
    num_hops=1,
    edge_index=edge_index,
    relabel_nodes=True,
    flow='target_to_source',
    directed=True,
)

ptensor(subset)
ptensor(edge_index)
ptensor(mapping)
ptensor(edge_mask)