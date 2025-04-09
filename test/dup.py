import torch

sums = torch.zeros(10, dtype=torch.int64)

nums = torch.tensor([2, 2, 2], dtype=torch.int64)
chosen = torch.tensor([2, 2, 2], dtype=torch.int64)


sums.scatter_add_(
    dim=0,
    index=chosen,
    src=nums,
)

print(sums)


mat = torch.tensor([2, 2, 2, 1, 1, 1], dtype=torch.int64)

print(mat.view(2, -1).transpose_(0, 1))
print(mat.view(-1, 2))