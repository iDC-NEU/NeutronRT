import torch
from torch_sparse import SparseTensor

# Example edge_index
edge_index = torch.tensor([[0, 1, 1, 2, 2],
                           [1, 0, 2, 1, 3]],
                           dtype=torch.long,
                           )

# Convert edge_index to SparseTensor
sparse_tensor = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(4, 4))

# Convert to CSR and CSC formats
rowptr, col, _ = sparse_tensor.csr()
colptr, row, _ = sparse_tensor.csc()

# Example: Find out-edges (CSR format) for node 1
node_id = 3
out_edges = rowptr[node_id:node_id+2]
print(f"Out-edges for node {node_id}: {out_edges}")

# Example: Find in-edges (CSC format) for node 1
in_edges = colptr[node_id:node_id+2]
print(f"In-edges for node {node_id}: {in_edges}")

# To get the actual edge indices, you can use:
out_edge_indices = col[rowptr[node_id]:rowptr[node_id+1]]
in_edge_indices = row[colptr[node_id]:colptr[node_id+1]]

print(f"Out-edge indices for node {node_id}: {out_edge_indices}")
print(f"In-edge indices for node {node_id}: {in_edge_indices}")