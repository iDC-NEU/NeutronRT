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
argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
argparser.add_argument('--sizes', type=str, default='-1,-1')
argparser.add_argument('--use-ratio', default=False, action='store_true')
argparser.add_argument('--emb-host-ratio', type=str, default='1,0')
argparser.add_argument('--emb-cache-ratio', type=str, default='1,1')
argparser.add_argument('--emb-host-size', type=str, default='0,12') # xGB
argparser.add_argument('--emb-cache-size', type=str, default='1,1')
argparser.add_argument('--task', type=str, default='none', 
                       choices=['link_pred', 'node', 'none'])
args = argparser.parse_args()

random.seed(0)
torch.manual_seed(0)

import resource
mem_limit = 450 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

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
data_in_gpu = data['data_in_gpu']

# Define PMA graph
print('Init graph ...', file=sys.stderr)
graph_csc = data['graph_csc']
graph_csr = data['graph_csr']
graph = PMAGraph(data_num_nodes, graph_csc, graph_csr, device, data['num_edges'])
data_degree = graph_csr[0][1:] - graph_csr[0][:-1]
_, topk_degree = torch.sort(data_degree, descending=True)

if 'x' in data:
    x = data['x'].pin_memory()
else:
    x = torch.rand((data_num_nodes, num_features), dtype=torch.float32, pin_memory=True)

update_lists = data['update_lists']
graph_coo = None if 'graph_coo' not in data else data['graph_coo']

task = Downstream(args.task, args.model, args.dataset, 
                  [num_features, args.num_hiddens, num_classes], sizes, device, 
                  data['update_labels'] if args.task == 'node' else None)
if args.task == 'none' and args.dataset == 'ogbn-papers100M':
    init_embs = None
else:
    init_embs, _ = task.init_embs(x, graph_coo, data_num_nodes, 
                                  batch_size_sample, args.dataset != 'GDELT')
dataset = dataset.close()
data = None

def get_ratio(size_item, num_dim):
    num_size = size_item * 1024 * 1024 * 1024 // (4 * num_dim)
    if num_size < data_num_nodes:
        num_item = num_size / data_num_nodes
    else:
        num_item = 1
    return num_item

# Define GPU cache 
print('Init cache ...', file=sys.stderr)
data_cache = [] # embedding cache
num_item_host = [float(ra) for ra in args.emb_host_ratio.split(',')] # todo: capacity
num_item_dev = [float(ra) for ra in args.emb_cache_ratio.split(',')]
# storing all embeddings of x0 and x_n on host for n-layer model
if not args.use_ratio:
    num_dev_size = [int(ra) for ra in args.emb_cache_size.split(',')]
    num_host_size = [int(ra) for ra in args.emb_host_size.split(',')]
    for i in range(0, num_model_layers):
        if i == 0:
            num_item_dev[i] = get_ratio(num_dev_size[i], num_features)
            num_item_host[i] = 1
        else:
            num_item_dev[i] = get_ratio(num_dev_size[i], args.num_hiddens)
            num_item_host[i] = get_ratio(num_host_size[i], args.num_hiddens)

for i in range(num_model_layers):
    num_dev = int(num_item_dev[i] * data_num_nodes)
    if  i == 0:
        num_dim = num_features
        num_host = data_num_nodes
        host_key = torch.arange(data_num_nodes, device=device)
        dev_key, _ = torch.sort(topk_degree[:num_dev])
        emb = x
        cache_emb = x[dev_key].to(device)
    else:
        num_dim = args.num_hiddens
        num_host = int(num_item_host[i] * data_num_nodes)
        dev_key, _ = torch.sort(topk_degree[:num_dev])
        if num_item_host[i] >= 1:
            host_key = torch.arange(data_num_nodes)
        else:
            if num_dev + num_host > data_num_nodes:
                num_host = data_num_nodes - num_dev
            host_key, _ = torch.sort(topk_degree[num_dev:num_dev+num_host])
        if init_embs is None:
            emb = torch.rand((host_key.size(0), num_dim), dtype=torch.float32, pin_memory=True)
            cache_emb = torch.rand((dev_key.size(0), num_dim), dtype=torch.float32, device=device)
        else:
            assert num_dim == init_embs[i-1].size(1)
            emb = init_embs[i-1][host_key].pin_memory()
            cache_emb = init_embs[i-1][dev_key].to(device)
    data_cache.append(GPUCache(data_num_nodes, host_key, emb, 
                                dev_key, cache_emb, device, data_in_gpu))

print(num_item_host, num_item_dev)
if init_embs is None:
    final_emb = torch.rand((data_num_nodes, num_classes), dtype=torch.float32, device='cpu')
else:
    final_emb = init_embs[-1].pin_memory()

sample_in_gpu = True
if args.dataset == 'twitter':
    sample_in_gpu = False
    if num_item_host[-1] <= 0:
        batch_size_sample = 512

update_graph_times = []
update_emb_times = []
inference_times = []
sample_times = []
transfer_times = []
compute_times = []
replace_times = []

list_acc = []
# list_f1 = []
# list_f1_w = []

print('Start running ...', file=sys.stderr)
for i in range(num_updates):
    insert_edges = update_lists[i][0]
    delete_edges = update_lists[i][1]

    sample_time = 0
    compute_time = 0
    transfer_time = 0
    replace_time = 0

    update_start = time.perf_counter()

    insert_edges = graph.insert_edges(insert_edges)
    delete_edges = graph.delete_edges(delete_edges)

    update_end = time.perf_counter()
    update_graph_times.append(update_end - update_start)

    infer_start = time.perf_counter()

    insert_edges = insert_edges.to(device)
    delete_edges = delete_edges.to(device)

    sample_start = time.perf_counter()

    com_emb_keys =[]    
    for layer_i in range(0, num_model_layers):
        if layer_i == 0:
            update_edge_index = torch.cat([delete_edges, insert_edges], dim=1)
            curr_com_emb_keys = update_edge_index[1].unique()
        else:
            curr_com_emb_keys = graph.next_hops(curr_com_emb_keys)
        if sample_in_gpu:
            com_emb_keys.append(curr_com_emb_keys)
        else:
            com_emb_keys.append(curr_com_emb_keys.cpu())

    if not sample_in_gpu:
        torch.cuda.empty_cache()

    relations = [None] * num_model_layers
    for layer_i in range(num_model_layers - 1, -1, -1):
        dst = com_emb_keys[layer_i].to(device)
        dst_indices, src = graph.sample_all_nodes(dst) # sorted
        if layer_i != 0:
            src_type = data_cache[layer_i].check_state(src) # 1 in gpu 0 in cpu -1 not in host 
            dst_type = data_cache[layer_i].check_state(dst)
            tmp_last_keys = torch.unique(torch.cat([com_emb_keys[layer_i-1].to(device), src[src_type < 0], dst[dst_type < 0]]))
            if not sample_in_gpu:
                tmp_last_keys = tmp_last_keys.cpu()
            com_emb_keys[layer_i-1] = tmp_last_keys
        if sample_in_gpu:
            relations[layer_i] = (src, dst_indices)
        else:
            relations[layer_i] = (src.cpu(), dst_indices.cpu())

    sample_time += time.perf_counter() - sample_start

    last_com_emb_values = None
    for layer_i in range(num_model_layers):
        curr_com_emb_keys = com_emb_keys[layer_i]
        if layer_i < num_model_layers - 1:
            curr_com_emb_values = torch.zeros((curr_com_emb_keys.size(0), data_cache[layer_i+1].num_feat), 
                        dtype=torch.float32, pin_memory=(not data_in_gpu), 
                        device=(device if data_in_gpu else 'cpu'))

        src_nodes, dst_indices = relations[layer_i]
        dst_indices = dst_indices.to(device)
        if layer_i > 0:
            last_emb_key = com_emb_keys[layer_i-1].to(device)
        next_start = 0
        while next_start < curr_com_emb_keys.size(0):
            sample_start = time.perf_counter()

            next_end = min(next_start + batch_size_sample, curr_com_emb_keys.size(0))
            next_nodes = curr_com_emb_keys[next_start:next_end].to(device)

            selected_src = src_nodes[dst_indices[next_start]:dst_indices[next_end]].to(device)

            # neibors
            unique_src, src_idx, src_counts = torch.unique(selected_src, return_inverse=True, return_counts=True)
            sample_end = time.perf_counter()

            # neibors
            src_emb, src_miss_index = data_cache[layer_i].query(unique_src, src_counts)
            if layer_i > 0 and torch.any(src_miss_index):
                com_src_index = torch.searchsorted(last_emb_key, unique_src[src_miss_index])
                if last_com_emb_values.is_cuda:
                    src_emb[src_miss_index] = last_com_emb_values[com_src_index]
                else:
                    src_emb[src_miss_index] = last_com_emb_values[com_src_index.cpu()].to(device)
                # if not torch.equal(unique_src[src_miss_index], last_emb_key[com_src_index]):
                #     print('err')
            
            dst_idx = dst_indices[next_start:next_end+1] - dst_indices[next_start]
            # target
            new_x_target, dst_miss_index = data_cache[layer_i].query(next_nodes)
            if layer_i > 0 and torch.any(dst_miss_index):
                com_dst_index = torch.searchsorted(last_emb_key, next_nodes[dst_miss_index])
                if last_com_emb_values.is_cuda:
                    new_x_target[dst_miss_index] = last_com_emb_values[com_dst_index]
                else:
                    new_x_target[dst_miss_index] = last_com_emb_values[com_dst_index.cpu()].to(device)
                # if not torch.equal(next_nodes[dst_miss_index], last_emb_key[com_dst_index]):
                #     print('err')

            transfer_end = time.perf_counter()

            csr_value = torch.ones_like(src_idx, dtype=torch.float32)
            
            if args.model == 'gcn':
                deg_row = graph.in_degree[unique_src].to(torch.float32)
                deg_row = deg_row.pow_(-0.5)
                deg_row.masked_fill_(deg_row == float('inf'), 0.)
                deg_row = deg_row[src_idx]
                deg_col = graph.in_degree[next_nodes].to(torch.float32)
                deg_col = deg_col.pow_(-0.5)
                deg_col.masked_fill_(deg_col == float('inf'), 0.)
                col_index = ptr2index(dst_idx)
                deg_col = deg_col[col_index]
                csr_value = deg_row * csr_value * deg_col

            csr = torch.sparse_csr_tensor(dst_idx, src_idx, csr_value, 
                size=(next_end - next_start, src_emb.size(0)),
                device=device)
            aggr_out = spmm(csr, src_emb, task.model.get_aggr(layer_i))

            update_y = task.model.layer_wise_partly(layer_i, aggr_out, new_x_target)
            compute_end = time.perf_counter()

            if layer_i < num_model_layers - 1:
                data_cache[layer_i+1].update(next_nodes, update_y)
                if curr_com_emb_values.is_cuda:
                    curr_com_emb_values[next_start:next_end] = update_y
                else:
                    curr_com_emb_values[next_start:next_end] = update_y.to(device='cpu', non_blocking=True)
            else:
                final_emb[next_nodes] = update_y.to(device='cpu', non_blocking=True)
            next_start = next_end

            sample_time += sample_end - sample_start
            transfer_time += transfer_end - sample_end
            compute_time += compute_end - transfer_end
        
        # may async
        replace_start = time.perf_counter()
        if num_item_dev[layer_i] < 1 and num_item_host[layer_i] > 0:
            if layer_i == 0 or num_item_host[layer_i] >= 1:
                if num_item_dev[layer_i] > 0:
                    data_cache[layer_i].replace(True)
            else:
                data_cache[layer_i].replace(False, 
                            [last_emb_key], 
                            [last_com_emb_values])
        replace_time += time.perf_counter() - replace_start

        if not curr_com_emb_values.is_cuda:
            torch.cuda.synchronize(device)
        if layer_i < num_model_layers - 1:
            last_com_emb_values = curr_com_emb_values
    
    torch.cuda.synchronize(device)
    update_emb_end = time.perf_counter()

    output = task.test(i, final_emb, args.task == 'link_pred')
    infer_end = time.perf_counter()

    print(f'Epoch {i}, edge size {graph.num_edges} ')
    if args.task == 'link_pred':
        list_acc.append(output)
        print(f'    AUC: {output}')
    else:
        # acc, f1, f1_w = output
        # print(f'    ACC: {acc}, F1 {f1}, F1_w {f1_w}')
        print(f'    ACC: {output}')
        list_acc.append(output)
        # list_f1.append(f1)
        # list_f1_w.append(f1_w)

    inference_times.append(infer_end - infer_start)
    update_emb_times.append(update_emb_end - infer_start)
    sample_times.append(sample_time)
    transfer_times.append(transfer_time)
    compute_times.append(compute_time)
    replace_times.append(replace_time)
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
           sample_times, transfer_times, replace_times,
           compute_times, 
           list_acc]:
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    print(f'{retrieve_name(ls)[0]}: {np.mean(ls, axis=0)}') 
    for i in ls:
        print(i, end=',')
    print()
