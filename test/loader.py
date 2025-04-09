import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import DataLoader, Dataset

# 创建一个示例图
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],  # 起始节点
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)  # 目标节点
x = torch.randn(4, 16)  # 节点特征
data = Data(x=x, edge_index=edge_index)

# 定义数据集
class KHopSubgraphDataset(Dataset):
    def __init__(self, node_indices, k, edge_index):
        self.node_indices = node_indices
        self.k = k
        self.edge_index = edge_index

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = self.node_indices[idx]
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.k, self.edge_index)
        return subset, sub_edge_index, mapping, edge_mask

# 定义要提取子图的节点
node_indices = torch.tensor([0, 1, 2, 3])

# 创建 DataLoader
dataset = KHopSubgraphDataset(node_indices, k=2, edge_index=edge_index)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

# 遍历 DataLoader 并合并结果
results = []
for batch in dataloader:
    results.append(batch)

# 合并所有结果
merged_results = {
    "subsets": torch.cat([r[0] for r in results]),
    "sub_edge_indices": torch.cat([r[1] for r in results], dim=1),
    "mappings": torch.cat([r[2] for r in results]),
    "edge_masks": torch.cat([r[3] for r in results])
}

print("合并后的结果:")
print(merged_results)