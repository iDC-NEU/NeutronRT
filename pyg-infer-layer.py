import argparse
import os
import random
import time
import numpy as np

import torch
from torch_geometric.utils import to_torch_csr_tensor, to_torch_coo_tensor
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset

from lib.pma_graph import *
from lib.gpu_cache import *

from model import SAGE,GAT,GCN

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=5)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--test-sizes', type=str, default='-1,-1')
argparser.add_argument('--compute-type', type=str, default="gpu")
args = argparser.parse_args()

sizes = [int(size) for size in args.sizes.split(',')]
test_sizes = [int(size) for size in args.test_sizes.split(',')]

path = './data/dataset'
os.makedirs(path, exist_ok=True)
num_model_layers = len(sizes)
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

orig_edge_index = to_torch_coo_tensor(data.edge_index).indices().type(torch.int64)
x = data.x.type(torch.float32).pin_memory()
y = data.y.view(-1).to(torch.int64)
data.x = x
data.y = y

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
select_index = set(range(orig_edge_index.size()[-1]))

for i in range(args.num_epochs):
    random.seed(i)
    index = random.sample(select_index, args.batch_size)
    pos = len(index) // 2
    delete_list.append(set(index[:pos]))
    insert_list.append(set(index[pos:]))
    select_index -= insert_list[i]

delete_list = delete_list[::-1]
insert_list = insert_list[::-1]

select_index_list = list(select_index)
select_index_list.sort()
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
data.edge_index = edge_index

embeddings = []
embeddings.append(x)
for i in range(num_model_layers):
    if i == num_model_layers - 1:
        emb =  torch.ones((data.num_nodes, num_classes), dtype=torch.float32, pin_memory=True)
    else:
        emb = torch.ones((data.num_nodes, args.num_hiddens), dtype=torch.float32, pin_memory=True)
    embeddings.append(emb)

update_graph_times = []
inference_times = []
sample_times = []
transfer_times = []
compute_times = []

for i in range(args.num_epochs):
    update_index = list(insert_list[i] | delete_list[i])
    update_edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(update_index))
    insert_edges = torch.index_select(orig_edge_index, 1, torch.tensor(list(insert_list[i])))
    delete_edges = list(delete_list[i])
    delete_edges.sort()
    delete_edges = torch.index_select(orig_edge_index, 1, torch.tensor(delete_edges))

    update_start = time.perf_counter()
    edge_index = data.edge_index
    remove_mask = torch.ones_like(edge_index[0])
    graph_csr = to_torch_csr_tensor(edge_index)
    csr_indexs = graph_csr.crow_indices()
    csr_rows = graph_csr.col_indices()
    for i_edges in range(len(delete_edges[0])):
        src = delete_edges[0][i_edges]
        dst = delete_edges[1][i_edges]
        for j_edges in range(csr_indexs[src], csr_indexs[src+1]):
            if csr_rows[j_edges] == dst:
                remove_mask[j_edges] = 0
    edge_index = edge_index[:, remove_mask == 1]

    edge_index = torch.cat([edge_index, insert_edges], dim=1)
    # _, sorted_ids = torch.sort(edge_index[0])
    # edge_index = edge_index[:, sorted_ids]
    data.edge_index = edge_index
    update_end = time.perf_counter()
    update_graph_times.append(update_end - update_start)

    infer_start = time.perf_counter()
    graph_csr = to_torch_csr_tensor(data.edge_index)
    csr_indexs = graph_csr.crow_indices()
    csr_rows = graph_csr.col_indices()

    unique_nodes = update_edge_index[1].unique()
    for layer_i in range(num_model_layers):
        if layer_i != 0:
            next_levels = [unique_nodes]
            for node in unique_nodes:
                next_levels.append(csr_rows[csr_indexs[node]:csr_indexs[node+1]])
            unique_nodes = torch.cat(next_levels).unique()

        loader = NeighborSampler(data.edge_index,
            node_idx=unique_nodes,
            sizes=[-1],
            batch_size=batch_size_sample,
            shuffle=False,
            num_workers=args.num_workers,
        )

        sample_start = time.perf_counter()
        other_time = 0
        for step, (batch_size, ids, adjs) in enumerate(loader):
            transfer_start = time.perf_counter()
            update_x = embeddings[layer_i][ids].to(device)
            sampled_edge_index = adjs[0].to(device)
            transfer_end = time.perf_counter()

            update_y = model.layer_wise(layer_i, update_x, sampled_edge_index, batch_size)
            compute_end = time.perf_counter()

            next_nodes = ids[:batch_size]
            embeddings[layer_i+1][next_nodes] = update_y.cpu()

            transfer_time += transfer_end - transfer_start
            compute_time += compute_end - transfer_end
            other_time += time.perf_counter() - transfer_start

        sample_time = time.perf_counter() - sample_start - other_time

    infer_end = time.perf_counter()
    inference_times.append(infer_end - infer_start)
    sample_times.append(sample_time)
    transfer_times.append(transfer_time)
    compute_times.append(compute_time)

import inspect

def retrieve_name(var):
    '''
    utils:
    get back the name of variables
    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


for ls in [update_graph_times, inference_times, sample_times, transfer_times, compute_times]:
    print(f'{retrieve_name(ls)[0]}: {np.mean(ls)}')
    for i in ls:
        print(i, end=',')
    print()
