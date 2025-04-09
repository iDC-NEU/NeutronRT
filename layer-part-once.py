import argparse
import os
import random
import time
import numpy as np
import sys

import torch
from torch_geometric.utils import spmm
from torch_geometric.utils.sparse import ptr2index

from lib.pma_graph import PMAGraph
from lib.gpu_cache import GPUCache
from lib.dataset import GraphDataset

from task import Downstream

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=1)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--model', type=str, default="sage",
                       choices=['sage', 'gcn', 'gin'])
argparser.add_argument('--dataset', type=str, default='ogbl-collab')
argparser.add_argument('--sizes', type=str, default='-1,-1')
argparser.add_argument('--task', type=str, default='none', 
                       choices=['link_pred', 'node', 'none'])
args = argparser.parse_args()

random.seed(0)
torch.manual_seed(0)

sizes = [int(size) for size in args.sizes.split(',')]

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)

path = './data/dataset'
os.makedirs(path, exist_ok=True)
num_model_layers = len(sizes)
batch_size_sample = 1024

num_updates = args.num_epochs
if args.task == 'link_pred':
    args.num_epochs += 1
dataset = GraphDataset(args.dataset, path, args.batch_size, args.num_epochs)

data = dataset.get_data(args.task == 'none')
num_features = data['num_features']
num_classes = data['num_classes']
if args.task == 'link_pred':
    num_classes = 64
data_num_nodes = data['num_nodes']
data_in_gpu = True

# Define PMA graph
print('Init graph ...', file=sys.stderr)
graph_csc = data['graph_csc']
graph_csr = data['graph_csr']
graph = PMAGraph(data_num_nodes, graph_csc, graph_csr, device, data['num_edges'])

if 'x' in data:
    x = data['x'].pin_memory()
else:
    x = torch.rand((data_num_nodes, num_features), dtype=torch.float32, pin_memory=True)

update_lists = data['update_lists']
graph_coo = None if 'graph_coo' not in data else data['graph_coo']

task = Downstream(args.task, args.model, args.dataset, 
                  [num_features, args.num_hiddens, num_classes], sizes, device, 
                  data['update_labels'] if args.task == 'node' else None)
init_embs, init_aggrs = task.init_embs(x, graph_coo, data_num_nodes, batch_size_sample, args.dataset != 'GDELT')

data_cache = [] # embedding cache
aggrs_cache = [] # aggregated embedding cache
for i in range(num_model_layers):
    num_dev = data_num_nodes
    dev_key = torch.arange(num_dev, dtype=torch.int64)
    if i == 0:
        num_dim = num_features
        num_host = data_num_nodes
        host_key = torch.arange(num_host, dtype=torch.int64, device=device)
        emb = x
        cache_emb = x[dev_key].to(device)
    else:
        num_dim = args.num_hiddens
        host_key = torch.empty(0, dtype=torch.int64, device=device)
        emb = torch.empty((0, num_dim), dtype=torch.float32, pin_memory=True)
        cache_emb = init_embs[i-1][dev_key].to(device)
    aggr_dev = data_num_nodes
    aggr_key = torch.arange(aggr_dev, dtype=torch.int64)
    cache_aggr = init_aggrs[i][aggr_key].to(device)
    aggr_host_key = torch.empty(0, dtype=torch.int64, device=device)
    aggr = torch.empty((0, num_dim), dtype=torch.float32, pin_memory=True)
    data_cache.append(GPUCache(data_num_nodes, host_key, emb, 
                                dev_key, cache_emb, device, data_in_gpu))
    aggrs_cache.append(GPUCache(data_num_nodes, aggr_host_key, aggr,
                                 aggr_key, cache_aggr, device, data_in_gpu))

final_emb = init_embs[-1].to(device)

update_graph_times = []
update_emb_times = []
inference_times = []
sample_times = []
transfer_times = []
compute_times = []

list_acc = []

print('Start running ...', file=sys.stderr)
for i in range(num_updates):
    insert_edges = update_lists[i][0]
    delete_edges = update_lists[i][1]

    sample_time = 0
    compute_time = 0
    transfer_time = 0

    update_start = time.perf_counter()

    last_in_degree = graph.clone_degree()
    insert_edges = graph.insert_edges(insert_edges, True)
    delete_edges = graph.delete_edges(delete_edges)

    update_end = time.perf_counter()
    update_graph_times.append(update_end - update_start)

    infer_start = time.perf_counter()

    insert_edges = insert_edges.to(device)
    delete_edges = delete_edges.to(device)
    dst_last_layer = None
    next_old_x = None

    for layer_i in range(num_model_layers):

        sample_start = time.perf_counter()
        src_nodes, dst_nodes, dst_indices, o_type, dst_counts, dst_expend_nodes = graph.get_next_hops(insert_edges, delete_edges, dst_last_layer)
        sample_end = time.perf_counter()
        
        # cache query old embedding -> get ahead with sorted ids -> binary search (searchsorted)
        if layer_i == 0:
            old_key = torch.unique(torch.cat([insert_edges[0], delete_edges[0]]))
            affect_old_x, _ = data_cache[layer_i].query(old_key, 0)
            new_key = old_key
            affect_new_x = affect_old_x
        else:
            (old_key, affect_old_x) = next_old_x
            new_key = torch.unique(torch.cat([insert_edges[0], dst_last_layer]))
            affect_new_x, _ = data_cache[layer_i].query(new_key)

        if layer_i + 1 < num_model_layers:
            next_layer_i = layer_i + 1
            next_old_key = torch.unique(torch.cat([delete_edges[0], dst_nodes]))
            next_affect_old_x, _ = data_cache[next_layer_i].query(next_old_key, 0)
            next_old_x = (next_old_key, next_affect_old_x)

        transfer_end = time.perf_counter()
        sample_time += sample_end - sample_start
        transfer_time += transfer_end - sample_end

        next_start = 0
        unchanged = torch.zeros_like(dst_nodes, dtype=torch.bool, device=device)
        while next_start < dst_nodes.size(0):
            sample_start = time.perf_counter()

            next_end = min(next_start + batch_size_sample, dst_nodes.size(0))
            next_nodes = dst_nodes[next_start:next_end]
            nodes_counts = dst_counts[next_start:next_end]
            selected_src = src_nodes[dst_indices[next_start]:dst_indices[next_end]]
            selected_type = o_type[dst_indices[next_start]:dst_indices[next_end]]
            
            # remap src nodes -> mapped_src load_type (0 load all 1 only load new -1 only load old)
            #    remap all remap src + remap dst (next_nodes, for compute)
            old_select_index = selected_type <= 0
            new_select_index = selected_type >= 0
            tmp_old = selected_src[old_select_index]
            tmp_new = selected_src[new_select_index]
            old_mapped_ids = torch.empty((2, tmp_old.size(0)), dtype=torch.int64, device=device)
            new_mapped_ids = torch.empty((2, tmp_new.size(0)), dtype=torch.int64, device=device)
            torch.searchsorted(old_key, tmp_old, out=old_mapped_ids[1])
            torch.searchsorted(new_key, tmp_new, out=new_mapped_ids[1])
            
            sample_end = time.perf_counter()
            
            # old aggreated embedding
            old_ax, _ = aggrs_cache[layer_i].query(next_nodes)
            new_x_target, _ = data_cache[layer_i].query(next_nodes)
            
            transfer_end = time.perf_counter()

            compute_type = task.model.get_aggr(layer_i)
            if compute_type == 'mean' or compute_type == 'sum' or compute_type == 'add':
                nodes_coo_src = dst_expend_nodes[dst_indices[next_start]:dst_indices[next_end]] - next_start
                torch.masked_select(nodes_coo_src, old_select_index, out=old_mapped_ids[0])
                torch.masked_select(nodes_coo_src, new_select_index, out=new_mapped_ids[0])
                old_coo_value = torch.ones_like(old_mapped_ids[0], dtype=torch.float32)
                new_coo_value = torch.ones_like(new_mapped_ids[0], dtype=torch.float32)
                if args.model == 'gcn':
                    old_deg_row = last_in_degree[tmp_old].to(torch.float32)
                    old_deg_row = old_deg_row.pow_(-0.5)
                    old_deg_row.masked_fill_(old_deg_row == float('inf'), 0.)
                    
                    deg_col = last_in_degree[next_nodes].to(torch.float32)
                    deg_col = deg_col.pow_(-0.5)
                    deg_col.masked_fill_(deg_col == float('inf'), 0.)
                    old_deg_col = deg_col[old_mapped_ids[0]]
                    old_coo_value = old_deg_row * old_coo_value * old_deg_col

                    new_deg_row = graph.in_degree[tmp_new].to(torch.float32)
                    new_deg_row = new_deg_row.pow_(-0.5)
                    new_deg_row.masked_fill_(new_deg_row == float('inf'), 0.)

                    new_deg_col = deg_col[new_mapped_ids[0]]
                    new_coo_value = new_deg_row * new_coo_value * new_deg_col
                old_ids_coo = torch.sparse_coo_tensor(old_mapped_ids, old_coo_value, 
                                                  (next_end - next_start, old_key.size(0)))
                new_ids_coo = torch.sparse_coo_tensor(new_mapped_ids, new_coo_value, 
                                                  (next_end - next_start, new_key.size(0)))
                new_ax = torch.zeros_like(old_ax, device=device, dtype=torch.float32)
                torch.sub(torch.sparse.mm(new_ids_coo, affect_new_x), 
                          torch.sparse.mm(old_ids_coo, affect_old_x), out=new_ax)
                if compute_type == 'mean':
                    sum_ax = torch.mul(old_ax, last_in_degree[next_nodes].view(-1, 1))
                    torch.add(new_ax, sum_ax, out=new_ax)
                    torch.div(new_ax, graph.in_degree[next_nodes].view(-1, 1), out=new_ax)
                else:
                    torch.add(new_ax, old_ax, out=new_ax)
            else:
                raise NotImplementedError

            update_y = task.model.layer_wise_partly(layer_i, new_ax, new_x_target)
            compute_end = time.perf_counter()

            torch.all(torch.isclose(new_ax, old_ax), dim=1, out=unchanged[next_start:next_end])
            aggrs_cache[layer_i].update(next_nodes, new_ax) # may async
            if layer_i < num_model_layers - 1:
                data_cache[layer_i+1].update(next_nodes, update_y) # may async
            else:
                final_emb[next_nodes] = update_y
            next_start = next_end

            sample_time += sample_end - sample_start
            transfer_time += transfer_end - sample_end
            compute_time += compute_end - transfer_end            

        dst_last_layer = dst_nodes[unchanged == False]
    
    update_emb_end = time.perf_counter()
    output = task.test(i, final_emb, args.task == 'link_pred')
    infer_end = time.perf_counter()
    graph.insert_edges_actual(insert_edges.cpu())

    print(f'Epoch {i}, edge size {graph.num_edges} ')
    if args.task == 'link_pred':
        list_acc.append(output)
        print(f'    AUC: {output}')
    else:
        print(f'    ACC: {output}')
        list_acc.append(output)

    inference_times.append(infer_end - infer_start)
    update_emb_times.append(update_emb_end - infer_start)
    sample_times.append(sample_time)
    transfer_times.append(transfer_time)
    compute_times.append(compute_time)
    print(f'Epoch {i} done in {inference_times[-1]} s.\n\tupdate emb {update_emb_times[-1]} s.', file=sys.stderr)

import inspect

def retrieve_name(var):
    '''
    utils:
    get back the name of variables
    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


for ls in [update_graph_times, update_emb_times, inference_times, 
           sample_times, transfer_times, 
           compute_times, list_acc]:
    print(f'{retrieve_name(ls)[0]}: {np.mean(ls)}')
    for i in ls:
        print(i, end=',')
    print()
