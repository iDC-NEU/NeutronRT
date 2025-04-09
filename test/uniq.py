import torch

# 假设 edge_index 是一个 (2, num_edges) 的张量
edge_index = torch.tensor([[0, 1, 1, 2, 1, 2],  # 起始节点
                           [1, 0, 2, 1, 2, 1]], dtype=torch.long)  # 目标节点

# 去重
# unique_edges, _ = torch.unique(edge_index, dim=1, return_inverse=True)

print("去重后的 edge_index:")
print(edge_index.unique(dim=1))