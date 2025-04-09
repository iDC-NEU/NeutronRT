import torch

key = torch.arange(100, dtype=torch.int64)
edge_index = torch.tensor([[0, 1, 2, 3, 4],
                           [1, 2, 3, 4, 5]])

src, dst = edge_index

ss = torch.searchsorted(key, src, side='right')

print(ss)