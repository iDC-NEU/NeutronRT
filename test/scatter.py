import torch
from torch_scatter import scatter

# 源张量
src = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

# 索引张量
index = torch.tensor([0, 1, 0])

# 使用 scatter 进行求和操作
out = scatter(src, index, dim=0, reduce='sum')

print(out)