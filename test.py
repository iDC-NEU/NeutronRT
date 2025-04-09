from lib.pma_graph import *
from lib.gpu_cache import *





import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, add_random_edge

edge_index = torch.tensor([[0, 2],
                           [1, 0]], dtype=torch.long)

# x = torch.tensor([[-1], [0], [1]], dtype=torch.float).pin_memory()
x = torch.ones((3, 128), dtype=torch.float).pin_memory()

for i in range(3):
    x[i] *= i

data = Data(x=x, edge_index=edge_index)

p_row, col, _ = data.csr()

print(p_row[None], col[None])

cuda_id = 4

device = torch.device('cuda:{}'.format(cuda_id))

graph = pmaGraph.PMAGraph(data.num_nodes, 
                          data.num_edges, 
                          p_row[None], col[None], cuda_id)




graph.hello()



# cache = gpuDataCache.GPUDataCache(2, 128, data.num_nodes, cuda_id, x)

# print('111')

# out = torch.zeros((1, 128), dtype=torch.float32, device=device)

# keys = torch.tensor([1], dtype=torch.long, device=device)
# print(keys.size(0))
# ret = cache.query(keys, 
#             torch.tensor([2], dtype=torch.long, device=device), out)

# print(ret, out)

# cache.replace()

# out = torch.zeros((2, 128), dtype=torch.float32, device=device)

# keys = torch.tensor([1, 2], dtype=torch.int64, device=device)
# print(keys.size(0))
# ret = cache.query(keys, 
#             torch.tensor([2, 1], dtype=torch.int64, device=device), out)

# print(ret, out)

# x[1] *= 4

# keys = torch.tensor([1], dtype=torch.long, device=device)

# cache.update(keys)
# out = torch.zeros((1, 128), dtype=torch.float32, device=device)
# ret = cache.query(keys, 
#             torch.tensor([2], dtype=torch.long, device=device), out)

# print(ret, out)

# print(data.num_nodes, data.num_edges)

# # data.put_edge_index((torch.tensor([1]), torch.tensor([2])))

# data.edge_index = torch.tensor([[0],
#                                 [1]], dtype=torch.long)

# # edge_index, _ = add_random_edge(edge_index)

# # data = Data(x=x, edge_index=edge_index)

# print(data.num_nodes, data.num_edges)
# p_row, col, _ = data.csr()

# print(p_row[None], col[None])

# edge_index, _ = dropout_edge(edge_index)

# data = Data(x=x, edge_index=edge_index)

# print(data.num_nodes, data.num_edges)

