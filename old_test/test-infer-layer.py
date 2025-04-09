import argparse
import os
import random

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_torch_csr_tensor, to_torch_csc_tensor, to_torch_coo_tensor
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset

from lib.pma_graph import *
from lib.gpu_cache import *

from model import SAGE,GAT,GCN

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=4)
argparser.add_argument('--num-epochs', type=int, default=1)
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--test-batch-size', type=int, default=1024)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--test-sizes', type=str, default='-1,-1')
# argparser.add_argument('--sizes', type=str, default='10,10,10')
# argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
# argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
# argparser.add_argument('--buffer-size', type=float, default=1)
# argparser.add_argument('--fallback', type=int, default=0)
args = argparser.parse_args()


path = './data/dataset'
os.makedirs(path, exist_ok=True)
num_model_layers = 2
batch_size_sample = 1024

if args.dataset == 'reddit':
    dataset = Reddit(path)
    data = dataset[0]
    split_idx = {'train': data.train_mask.nonzero(as_tuple=False).view(-1), 
                 'valid': data.val_mask.nonzero(as_tuple=False).view(-1), 
                 'test': data.test_mask.nonzero(as_tuple=False).view(-1)}
elif args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(path, args.dataset)
    data = dataset[0]
    split_idx = {'train': data.train_mask.nonzero(as_tuple=False).view(-1), 
                 'valid': data.val_mask.nonzero(as_tuple=False).view(-1), 
                 'test': data.test_mask.nonzero(as_tuple=False).view(-1)}
else:
    dataset = PygNodePropPredDataset(args.dataset, path)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

num_features = dataset.num_features
num_classes = dataset.num_classes

sizes = [int(size) for size in args.sizes.split(',')]
test_sizes = [int(size) for size in args.test_sizes.split(',')]

orig_edge_index = to_torch_coo_tensor(data.edge_index).indices().type(torch.int64)
x = data.x.type(torch.float32).pin_memory()
y = data.y.view(-1).to(torch.int64)
data.x = x
# data.y = y

# Define model
if args.compute_type == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:%d' % args.gpu)
    torch.cuda.set_device(device)

def new_model():
    if args.model == 'sage':
        model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
    elif args.model == 'gcn':
        model = GCN(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), 
                    norm=True)
    elif args.model == 'gat':
        model = GAT(num_features, args.num_hiddens, num_classes, num_layers=len(sizes),
                    heads=4)
    else:
        raise NotImplementedError

    model = model.to(device)
    return model

model = new_model()
model.eval()

# Define edge update
delete_list = []
insert_list = []
remove_ahead = set()
select_index = range(orig_edge_index.size()[-1])

for i in range(args.num_epochs):
    random.seed(i)
    index = random.sample(select_index, args.batch_size)
    delete_list.append(set(index[:len(index) // 2]))
    index_set = set(index[len(index) // 2:])
    insert_list.append(index_set)
    remove_ahead |= index_set

select_index = set(select_index) - remove_ahead
select_index_list = list(select_index)
select_index_list.sort()
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
data.edge_index = edge_index

# Define PMA graph
embeddings = []
data_cache = []
num_item = 100
embeddings.append(x)
data_cache.append(gpuDataCache.GPUDataCache(num_item, x.size(1), x.size(0), args.gpu, x))
for i in range(num_model_layers):
    if i == num_model_layers - 1:
        emb =  torch.ones((data.num_nodes, num_classes), dtype=torch.float32).pin_memory()
    else:
        emb = torch.ones((data.num_nodes, args.num_hiddens), dtype=torch.float32).pin_memory()
    embeddings.append(emb)
    data_cache.append(gpuDataCache.GPUDataCache(num_item, emb.size(1), emb.size(0), args.gpu, emb))

graph_csc = to_torch_csc_tensor(data.edge_index)
graph_csr = to_torch_csr_tensor(data.edge_index)

graph = pmaGraph.PMAGraph(data.num_nodes, 
                          len(graph_csc.row_indices()), 
                          graph_csc.ccol_indices(), graph_csc.row_indices(), args.gpu)
t_graph = pmaGraph.PMAGraph(data.num_nodes, 
                          len(graph_csr.col_indices()), 
                          graph_csr.crow_indices(), graph_csr.col_indices(), args.gpu)

def graph_T(graph_):
    indice = torch.zeros((graph_.num_nodes() + 1), dtype=torch.int64, device=device)
    rows = torch.zeros((graph_.num_edges()), dtype=torch.int64, device=device)
    out = graph_.transpose(indice, rows)
    return out, indice, rows

def check_graph(graph_, data_):
    graph_csr = to_torch_csr_tensor(data_.edge_index)
    out, indice, rows = graph_T(graph_)
    print(out, 
          torch.equal(indice, graph_csr.crow_indices().to(device)), 
          torch.equal(rows, graph_csr.col_indices().to(device)))
    # for i in range(len(indice)):
    #     p_row = graph_csr.col_indices().to(device)[indice[i]:indice[i+1]]
    #     sorted_p, _ = torch.sort(p_row)
    #     q_row = rows[indice[i]:indice[i+1]]
    #     sorted_q, _ = torch.sort(q_row)
    #     if not torch.equal(sorted_p, sorted_q):
    #         print(i)
    # torch.cuda.empty_cache()

# check_graph(graph, data)

for i in range(args.num_epochs):
    select_index |= insert_list[i]
    select_index -= delete_list[i]
    select_index_list = list(select_index)
    select_index_list.sort()
    edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
    data.edge_index = edge_index

    update_index = list(insert_list[i] | delete_list[i])
    update_edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(update_index))
    update_nodes = [set() for i in range(num_model_layers)]

    for edge in update_edge_index[1]:
       update_nodes[0].add(int(edge)) # dst

    for i_node in insert_list[i]:
        # dst <- src
        graph.insert_edge(int(orig_edge_index[1][i_node]), int(orig_edge_index[0][i_node]), 0)
        t_graph.insert_edge(int(orig_edge_index[0][i_node]), int(orig_edge_index[1][i_node]), 0)
    for i_node in delete_list[i]:
        graph.delete_edge(int(orig_edge_index[1][i_node]), int(orig_edge_index[0][i_node]), 0)
        t_graph.delete_edge(int(orig_edge_index[0][i_node]), int(orig_edge_index[1][i_node]), 0)

    # check_graph(graph, data)
    
    csr_out, csr_indices, csr_rows = graph_T(graph)

    for j in range(1, num_model_layers):
        for node in update_nodes[j-1]:
            for edge_index in range(csr_indices[node], csr_indices[node+1]):
                edge = int(csr_rows[edge_index])
                update_nodes[j].add(edge)

    unique_nodes = update_edge_index[1].to(device=device).unique()
    recompute_nodes = set()
    for layer_i in range(num_model_layers):
        if layer_i != 0:
            # next_levels = [unique_nodes]
            # for node in unique_nodes:
            #     next_levels.append(csr_rows[csr_indices[node]:csr_indices[node+1]])
            # unique_nodes = torch.cat(next_levels).unique()
            unique_nodes = t_graph.next_hops(unique_nodes).unique()

        recompute_nodes |= update_nodes[layer_i]
        recompute_tensor = torch.tensor(list(recompute_nodes))
        recompute_tensor, _ = torch.sort(recompute_tensor)
        print("Layer:", layer_i, torch.equal(unique_nodes, recompute_tensor.to(device)))

        res_graph = []
        next_start = 0
        while next_start < unique_nodes.size(0):
            next_end = min(next_start + batch_size_sample, unique_nodes.size(0))
            next_nodes = unique_nodes[next_start:next_end]
            output = graph.sample_neighbor(next_nodes, -1)
            (ids, row_idx, col_idx) = output
            counts_ids = torch.bincount(col_idx)
            update_x = torch.zeros((ids.size(0), embeddings[layer_i].size(1)),
                                    dtype=torch.float32, device=device)
            ret = data_cache[layer_i].query(ids, counts_ids, update_x)
            if not ret:
                raise RuntimeError("Error: query failed")

            edge_index = torch.stack([col_idx, row_idx], dim=0)
            update_y = model.layer_wise(layer_i, update_x, edge_index, next_nodes.size(0))
            data_cache[layer_i+1].update(next_nodes, update_y) # may async
            next_start = next_end

            res_graph.append(output)

        # may async
        data_cache[layer_i].replace()


        loader = NeighborSampler(data.edge_index,
            node_idx=recompute_tensor,
            sizes=[-1],
            batch_size=batch_size_sample,
            shuffle=False,
            num_workers=args.num_workers,
        )

        for step, (batch_size, ids, adjs) in enumerate(loader):
            edge_index = adjs.edge_index

            (g_ids, g_row_idx, g_col_idx) = res_graph[step]

            sorted_ids, _ = torch.sort(ids)
            sorted_g_ids, _ = torch.sort(g_ids)

            pre_ids, _ = torch.sort(ids[:batch_size])
            pre_g_ids, _ = torch.sort(g_ids[:batch_size])

            print(torch.equal(g_ids[:batch_size], ids[:batch_size].to(device)), 
                  torch.equal(pre_g_ids, pre_ids.to(device)), 
                  torch.equal(sorted_g_ids, sorted_ids.to(device)), 
                  torch.equal(g_row_idx, edge_index[1].to(device)), 
                  torch.equal(g_col_idx, edge_index[0].to(device)))

