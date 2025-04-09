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
from lib.gpu_cache import GPUCache, check_part_compute_state
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
argparser.add_argument('--use-ratio', default=False, action='store_true')
argparser.add_argument('--emb-host-ratio', type=str, default='1,0')
argparser.add_argument('--emb-cache-ratio', type=str, default='1,1')
argparser.add_argument('--aggr-host-ratio', type=str, default='0,0')
argparser.add_argument('--aggr-cache-ratio', type=str, default='1,1')
argparser.add_argument('--emb-host-size', type=str, default='0,12')
argparser.add_argument('--emb-cache-size', type=str, default='1,1')
argparser.add_argument('--aggr-host-size', type=str, default='10,10')
argparser.add_argument('--aggr-cache-size', type=str, default='1,1')
argparser.add_argument('--task', type=str, default='none', 
                       choices=['link_pred', 'node', 'none'])
args = argparser.parse_args()

random.seed(0)
torch.manual_seed(0)

import resource
mem_limit = 490 * 1024 ** 3
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
    init_embs = init_aggrs = None
else:
    init_embs, init_aggrs = task.init_embs(x, graph_coo, data_num_nodes, batch_size_sample, args.dataset != 'GDELT')
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
        host_key = torch.arange(data_num_nodes, dtype=torch.int64)
        if num_item_dev[i] < 1:
            dev_key, _ = torch.sort(topk_degree[:num_dev])
        else:
            dev_key = torch.arange(num_dev, dtype=torch.int64)
        cache_emb = x[dev_key].to(device)
        emb = x
    else:
        num_dim = args.num_hiddens
        if num_item_dev[i] < 1:
            num_host = int(num_item_host[i] * data_num_nodes)
            dev_key, _ = torch.sort(topk_degree[:num_dev])
            if num_item_dev[i] < 1 and num_item_host[i] >= 1:
                host_key = torch.arange(data_num_nodes)
            else:
                if num_dev + num_host > data_num_nodes:
                    num_host = data_num_nodes - num_dev
                host_key, _ = torch.sort(topk_degree[num_dev:num_dev+num_host])
        else:
            num_host = 0
            dev_key = torch.arange(num_dev, dtype=torch.int64)
            host_key = torch.empty(0, dtype=torch.int64)
        if init_embs is None:
            emb = torch.rand((host_key.size(0), num_dim), dtype=torch.float32, pin_memory=True)
            cache_emb = torch.rand((dev_key.size(0), num_dim), dtype=torch.float32, device=device)
        else:
            assert num_dim == init_embs[i-1].size(1)
            emb = init_embs[i-1][host_key].pin_memory()
            cache_emb = init_embs[i-1][dev_key].to(device)
    data_cache.append(GPUCache(data_num_nodes, host_key, emb, 
                                dev_key, cache_emb, device, data_in_gpu))

aggrs_cache = [] # aggregated embedding cache
num_aggrs_host = [float(ra) for ra in args.aggr_host_ratio.split(',')]
num_aggrs_dev = [float(ra) for ra in args.aggr_cache_ratio.split(',')]
if not args.use_ratio:
    num_host_size = [int(ra) for ra in args.aggr_host_size.split(',')]
    num_dev_size = [int(ra) for ra in args.aggr_cache_size.split(',')]
    for i in range(num_model_layers):
        num_dim = args.num_hiddens
        if i == 0:
            num_dim = num_features
        num_aggrs_host[i] = get_ratio(num_host_size[i], num_dim)
        num_aggrs_dev[i] = get_ratio(num_dev_size[i], num_dim)

for i in range(num_model_layers):
    num_dim = args.num_hiddens
    if i == 0:
        num_dim = num_features
    num_dev = int(num_aggrs_dev[i] * data_num_nodes)
    if num_aggrs_dev[i] < 1:
        num_host = int(num_aggrs_host[i] * data_num_nodes)
        dev_key, _ = torch.sort(topk_degree[:num_dev])
        if num_aggrs_host[i] >= 1:
            host_key = torch.arange(data_num_nodes)
        else:
            if num_dev + num_host > data_num_nodes:
                num_host = data_num_nodes - num_dev
            host_key, _ = torch.sort(topk_degree[num_dev:num_dev+num_host])
    else:
        num_host = 0
        dev_key = torch.arange(num_dev, dtype=torch.int64)
        host_key = torch.empty(0, dtype=torch.int64)
    if init_aggrs is None:
        emb = torch.rand((host_key.size(0), num_dim), dtype=torch.float32, pin_memory=True)
        cache_emb = torch.rand((dev_key.size(0), num_dim), dtype=torch.float32, device=device)
    else:
        assert num_dim == init_aggrs[i].size(1)
        emb = init_aggrs[i][host_key].pin_memory()
        cache_emb = init_aggrs[i][dev_key].to(device)
    aggrs_cache.append(GPUCache(data_num_nodes, host_key, emb,
                                 dev_key, cache_emb, device, data_in_gpu))

print(num_item_host, num_item_dev, num_aggrs_host, num_aggrs_dev)
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

print('Start running ...', file=sys.stderr)
for i in range(num_updates):
    insert_edges = update_lists[i][0]
    delete_edges = update_lists[i][1]

    # ahead, part, full
    sample_time = [0, 0, 0]
    compute_time = [0, 0, 0]
    transfer_time = [0, 0, 0]
    replace_time = [0, 0, 0]

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

    sample_start = time.perf_counter()

    full_com_emb_keys = []
    part_com_emb_keys = []
    for layer_i in range(num_model_layers):
        src, mapped_dst, dst_indices, o_type, dst_counts = graph.next_affact_nodes(insert_edges, delete_edges, dst_last_layer)
        
        # check both dst and src at the same time
        # return bool for dst and src (src and type)
        can_part, need_src = check_part_compute_state(aggrs_cache[layer_i], data_cache[layer_i], mapped_dst, src, dst_indices, o_type)
        # can_part = torch.zeros_like(mapped_dst, dtype=torch.bool)
        # need_src = torch.zeros_like(src, dtype=torch.bool)
        p_dst = mapped_dst[can_part]
        p_counts = dst_counts[can_part]
        p_src = src[need_src]
        p_type = o_type[need_src]
        p_extend = torch.arange(p_counts.size(0), dtype=torch.int64, device=device).repeat_interleave(p_counts)
        p_indices = torch.zeros(p_counts.size(0) + 1, dtype=torch.int64, device=device)
        torch.cumsum(p_counts, 0, out=p_indices[1:])
        if layer_i == 0:
            old_key = None
        else:
            old_key = torch.unique(p_src[p_type <= 0])
        part_com_emb_keys.append({'dst': p_dst, 
                                  'src': p_src,
                                  'idx': p_indices, 
                                  'extend': p_extend,
                                  'type': p_type, 
                                  'old': old_key})
        full_com_dst = mapped_dst[can_part == False]
        aggrs_cache[layer_i].invalidate(full_com_dst)
        full_com_emb_keys.append(full_com_dst)
        dst_last_layer = mapped_dst
    
    relations = [None] * num_model_layers
    for layer_i in range(num_model_layers - 1, -1, -1):
        dst = full_com_emb_keys[layer_i]
        dst_contains = aggrs_cache[layer_i].check_state(dst, False)
        dst_indices, src = graph.sample_all_nodes(dst, dst_contains >= 0)
        if layer_i != 0:
            # 1 in gpu 0 in cpu -1 not in host
            src_type = data_cache[layer_i].check_state(src) # 1 in gpu 0 in cpu -1 not in host 
            dst_type = data_cache[layer_i].check_state(dst)
            part_src = part_com_emb_keys[layer_i]['src']
            part_src_type = data_cache[layer_i].check_state(part_src)
            part_dst = part_com_emb_keys[layer_i]['dst']
            part_dst_type = data_cache[layer_i].check_state(part_dst)
            # full_com_emb_keys[layer_i-1] = torch.unique(torch.cat([full_com_emb_keys[layer_i-1], 
            #     src[src_type < 0], dst[dst_type < 0], part_dst[part_dst_type < 0]]))
            full_com_emb_keys[layer_i-1] = torch.unique(torch.cat([full_com_emb_keys[layer_i-1], 
                                                        src[src_type < 0], 
                                                        dst[dst_type < 0], 
                                                        part_src[part_src_type < 0], 
                                                        part_dst[part_dst_type < 0]]))
            # TODO Exclude nodes partially updated

        relations[layer_i] = (src, dst_indices, dst_contains)

    sample_time[0] += time.perf_counter() - sample_start

    last_com_emb_values = None
    last_part_emb_values = None
    for layer_i in range(num_model_layers):
        # update partly
        sample_start = time.perf_counter()
        p_src = part_com_emb_keys[layer_i]['src']
        p_dst = part_com_emb_keys[layer_i]['dst']
        dst_indices = part_com_emb_keys[layer_i]['idx']
        p_type = part_com_emb_keys[layer_i]['type']
        p_extend = part_com_emb_keys[layer_i]['extend']
        if layer_i == 0:
            old_key = new_key = torch.unique(p_src)
        else:
            old_key = part_com_emb_keys[layer_i]['old']
            new_key = torch.unique(p_src[p_type >= 0])
        sample_end = time.perf_counter()
        
        # cache query old embedding -> get ahead with sorted ids -> binary search (searchsorted)
        if layer_i == 0:
            affect_old_x, test_miss = data_cache[layer_i].query(old_key)
            affect_new_x = affect_old_x
        else:
            affect_old_x = next_old_x
            # affect_new_x, test_miss = data_cache[layer_i].query(new_key)
            affect_new_x, src_miss_index = data_cache[layer_i].query(new_key)
            if torch.any(src_miss_index):
                com_src_index = torch.searchsorted(full_com_emb_keys[layer_i-1], new_key[src_miss_index])
                if last_com_emb_values.is_cuda:
                    affect_new_x[src_miss_index] = last_com_emb_values[com_src_index]
                else:
                    affect_new_x[src_miss_index] = last_com_emb_values[com_src_index.cpu()].to(device)

        if layer_i + 1 < num_model_layers:
            next_layer_i = layer_i + 1
            next_old_key = part_com_emb_keys[next_layer_i]['old']
            next_affect_old_x, test_miss = data_cache[next_layer_i].query(next_old_key, 0)
            next_old_x = next_affect_old_x

        transfer_end = time.perf_counter()
        sample_time[1] += sample_end - sample_start
        transfer_time[1] += transfer_end - sample_end

        # update partly computed aggreated embedding
        next_start = 0
        curr_aggr_values = torch.zeros((p_dst.size(0), aggrs_cache[layer_i].num_feat), 
                    dtype=torch.float32, pin_memory=(not data_in_gpu), 
                    device=(device if data_in_gpu else 'cpu'))
        # unchanged = torch.zeros_like(dst_nodes, dtype=torch.bool, device=device)

        if layer_i < num_model_layers - 1:
            part_curr_com_emb = torch.zeros((p_dst.size(0), data_cache[layer_i+1].num_feat), 
                    dtype=torch.float32, pin_memory=(not data_in_gpu), 
                    device=(device if data_in_gpu else 'cpu'))

        while next_start < p_dst.size(0):
            sample_start = time.perf_counter()

            next_end = min(next_start + batch_size_sample, p_dst.size(0))
            next_nodes = p_dst[next_start:next_end]
            selected_src = p_src[dst_indices[next_start]:dst_indices[next_end]]
            selected_type = p_type[dst_indices[next_start]:dst_indices[next_end]]
            
            # remap src nodes -> mapped_src load_type (0 load all 1 only load new -1 only load old)
            #    remap all remap src + remap dst (next_nodes, for compute)
            old_select_index = selected_type <= 0
            new_select_index = selected_type >= 0
            # tmp_old = selected_src[old_select_index]
            # tmp_new = selected_src[new_select_index]
            # old_mapped_ids = torch.empty((2, tmp_old.size(0)), dtype=torch.int64, device=device)
            # new_mapped_ids = torch.empty((2, tmp_new.size(0)), dtype=torch.int64, device=device)
            # torch.searchsorted(old_key, tmp_old, out=old_mapped_ids[1])
            # torch.searchsorted(new_key, tmp_new, out=new_mapped_ids[1])

            old_src_idx = torch.searchsorted(old_key, selected_src)
            new_src_idx = torch.searchsorted(new_key, selected_src)

            sample_end = time.perf_counter()
            
            # old aggreated embedding
            old_ax, test_miss = aggrs_cache[layer_i].query(next_nodes)
            # new_x_target, test_miss = data_cache[layer_i].query(next_nodes)

            # target
            new_x_target, target_miss = data_cache[layer_i].query(next_nodes)
            if layer_i > 0 and torch.any(target_miss):
                target_index = torch.searchsorted(full_com_emb_keys[layer_i-1], next_nodes[target_miss])
                if last_com_emb_values.is_cuda:
                    new_x_target[target_miss] = last_com_emb_values[target_index]
                else:
                    new_x_target[target_miss] = last_com_emb_values[target_index.cpu()].to(device)

            transfer_end = time.perf_counter()

            compute_type = task.model.get_aggr(layer_i)
            if compute_type == 'mean' or compute_type == 'sum' or compute_type == 'add':
                # nodes_coo_src = p_extend[dst_indices[next_start]:dst_indices[next_end]] - next_start
                # torch.masked_select(nodes_coo_src, old_select_index, out=old_mapped_ids[0])
                # torch.masked_select(nodes_coo_src, new_select_index, out=new_mapped_ids[0])
                # old_coo_value = torch.ones_like(old_mapped_ids[0], dtype=torch.float32)
                # new_coo_value = torch.ones_like(new_mapped_ids[0], dtype=torch.float32)
                cur_dst_indices = dst_indices[next_start:next_end+1] - dst_indices[next_start]
                old_csr_value = old_select_index.to(dtype=torch.float32)
                new_csr_value = new_select_index.to(dtype=torch.float32)

                if args.model == 'gcn':
                    # old_deg_row = last_in_degree[tmp_old].to(torch.float32)
                    # old_deg_row = old_deg_row.pow_(-0.5)
                    # old_deg_row.masked_fill_(old_deg_row == float('inf'), 0.)
                    
                    # deg_col = last_in_degree[next_nodes].to(torch.float32)
                    # deg_col = deg_col.pow_(-0.5)
                    # deg_col.masked_fill_(deg_col == float('inf'), 0.)
                    # old_deg_col = deg_col[old_mapped_ids[0]]
                    # old_coo_value = old_deg_row * old_coo_value * old_deg_col

                    # new_deg_row = graph.in_degree[tmp_new].to(torch.float32)
                    # new_deg_row = new_deg_row.pow_(-0.5)
                    # new_deg_row.masked_fill_(new_deg_row == float('inf'), 0.)

                    # new_deg_col = deg_col[new_mapped_ids[0]]
                    # new_coo_value = new_deg_row * new_coo_value * new_deg_col
                    old_deg_row = last_in_degree[selected_src].to(torch.float32)
                    old_deg_row = old_deg_row.pow_(-0.5)
                    old_deg_row.masked_fill_(old_deg_row == float('inf'), 0.)
                    
                    nodes_coo_src = p_extend[dst_indices[next_start]:dst_indices[next_end]] - next_start
                    deg_col = last_in_degree[next_nodes].to(torch.float32)
                    deg_col = deg_col.pow_(-0.5)
                    deg_col.masked_fill_(deg_col == float('inf'), 0.)
                    old_deg_col = deg_col[nodes_coo_src]
                    old_csr_value = old_deg_row * old_csr_value * old_deg_col

                    new_deg_row = graph.in_degree[selected_src].to(torch.float32)
                    new_deg_row = new_deg_row.pow_(-0.5)
                    new_deg_row.masked_fill_(new_deg_row == float('inf'), 0.)

                    new_deg_col = deg_col[nodes_coo_src]
                    new_csr_value = new_deg_row * new_csr_value * new_deg_col

                # old_ids_coo = torch.sparse_coo_tensor(old_mapped_ids, old_coo_value, 
                #                                   (next_end - next_start, old_key.size(0)))
                # new_ids_coo = torch.sparse_coo_tensor(new_mapped_ids, new_coo_value, 
                #                                   (next_end - next_start, new_key.size(0)))
                
                old_ids_csr = torch.sparse_csr_tensor(cur_dst_indices, old_src_idx, old_csr_value, 
                        size=(next_end - next_start, affect_old_x.size(0)),
                        device=device)
                new_ids_csr = torch.sparse_csr_tensor(cur_dst_indices, new_src_idx, new_csr_value, 
                        size=(next_end - next_start, affect_new_x.size(0)),
                        device=device)

                new_ax = torch.zeros_like(old_ax, device=device, dtype=torch.float32)

                # torch.sub(torch.sparse.mm(new_ids_coo, affect_new_x), 
                #           torch.sparse.mm(old_ids_coo, affect_old_x), 
                #           out=new_ax)
                torch.sub(torch.sparse.mm(new_ids_csr, affect_new_x), 
                          torch.sparse.mm(old_ids_csr, affect_old_x),
                          out=new_ax)
                if compute_type == 'mean':
                    sum_ax = torch.mul(old_ax, last_in_degree[next_nodes].view(-1, 1))
                    torch.add(new_ax, sum_ax, out=new_ax)
                    torch.div(new_ax, graph.in_degree[next_nodes].view(-1, 1), 
                              out=new_ax)
                else:
                    torch.add(new_ax, old_ax, out=new_ax)
            else:
                raise NotImplementedError

            # update_y = model.layer_wise_partly(layer_i, new_ax, new_x_target)
            update_y = task.model.layer_wise_partly(layer_i, new_ax, new_x_target)
            compute_end = time.perf_counter()
            # torch.all(torch.isclose(new_ax, old_ax), dim=1, out=unchanged[next_start:next_end])
            aggrs_cache[layer_i].update(next_nodes, new_ax) # may async
            # data_cache[layer_i+1].update(next_nodes, update_y) # may async
            if curr_aggr_values.is_cuda:
                curr_aggr_values[next_start:next_end] = new_ax
            else:
                curr_aggr_values[next_start:next_end] = new_ax.to(device='cpu', non_blocking=True)

            if layer_i < num_model_layers - 1:
                data_cache[layer_i+1].update(next_nodes, update_y) # may async
                if part_curr_com_emb.is_cuda:
                    part_curr_com_emb[next_start:next_end] = update_y
                else:
                    part_curr_com_emb[next_start:next_end] = update_y.to(device='cpu', non_blocking=True)
            else:
                final_emb[next_nodes] = update_y.to(device='cpu', non_blocking=True)
            next_start = next_end

            sample_time[1] += sample_end - sample_start
            transfer_time[1] += transfer_end - sample_end
            compute_time[1] += compute_end - transfer_end

        # dst_last_layer = dst_nodes[unchanged == False]

        # update full
        curr_full_keys = full_com_emb_keys[layer_i]
        if layer_i < num_model_layers - 1:
            curr_com_emb_values = torch.zeros((curr_full_keys.size(0), data_cache[layer_i+1].num_feat), 
                        dtype=torch.float32, pin_memory=(not data_in_gpu), 
                        device=(device if data_in_gpu else 'cpu'))
        curr_full_aggrs = torch.zeros((curr_full_keys.size(0), aggrs_cache[layer_i].num_feat), 
                        dtype=torch.float32, pin_memory=(not data_in_gpu), 
                        device=(device if data_in_gpu else 'cpu'))

        src_nodes, dst_indices, dst_contains = relations[layer_i]
        next_start = 0
        while next_start < curr_full_keys.size(0):
            sample_start = time.perf_counter()

            next_end = min(next_start + batch_size_sample, curr_full_keys.size(0))
            next_nodes = curr_full_keys[next_start:next_end]

            selected_src = src_nodes[dst_indices[next_start]:dst_indices[next_end]]

            # neibors
            unique_src, src_idx, src_counts = torch.unique(selected_src, return_inverse=True, return_counts=True)
            sample_end = time.perf_counter()

            # neibors
            src_emb, src_miss_index = data_cache[layer_i].query(unique_src, src_counts)
            if layer_i > 0 and torch.any(src_miss_index):
                com_src_index = torch.searchsorted(full_com_emb_keys[layer_i-1], unique_src[src_miss_index])
                if last_com_emb_values.is_cuda:
                    src_emb[src_miss_index] = last_com_emb_values[com_src_index]
                else:
                    src_emb[src_miss_index] = last_com_emb_values[com_src_index.cpu()].to(device)
            
            dst_idx = dst_indices[next_start:next_end+1] - dst_indices[next_start]
            # target
            new_x_target, dst_miss_index = data_cache[layer_i].query(next_nodes)
            if layer_i > 0 and torch.any(dst_miss_index):
                com_dst_index = torch.searchsorted(full_com_emb_keys[layer_i-1], next_nodes[dst_miss_index])
                if last_com_emb_values.is_cuda:
                    new_x_target[dst_miss_index] = last_com_emb_values[com_dst_index]
                else:
                    new_x_target[dst_miss_index] = last_com_emb_values[com_dst_index.cpu()].to(device)

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
            _, miss = aggrs_cache[layer_i].query(next_nodes, out=aggr_out, ignore=False)

            update_y = task.model.layer_wise_partly(layer_i, aggr_out, new_x_target)
            compute_end = time.perf_counter()

            aggrs_cache[layer_i].update(next_nodes, aggr_out, miss == False) 
            if curr_full_aggrs.is_cuda:
                curr_full_aggrs[next_start:next_end] = aggr_out
            else:
                curr_full_aggrs[next_start:next_end] = aggr_out.to(device='cpu', non_blocking=True)

            if layer_i < num_model_layers - 1:
                data_cache[layer_i+1].update(next_nodes, update_y)
                if curr_com_emb_values.is_cuda:
                    curr_com_emb_values[next_start:next_end] = update_y
                else:
                    curr_com_emb_values[next_start:next_end] = update_y.to(device='cpu', non_blocking=True)
            else:
                final_emb[next_nodes] = update_y.to(device='cpu', non_blocking=True)
            next_start = next_end

            sample_time[2] += sample_end - sample_start
            transfer_time[2] += transfer_end - sample_end
            compute_time[2] += compute_end - transfer_end
        
        # may async
        replace_start = time.perf_counter()
        if num_item_dev[layer_i] < 1:
            if layer_i == 0 or num_item_host[layer_i] >= 1:
                data_cache[layer_i].replace(True)
            else:
                data_cache[layer_i].replace(False, 
                            [full_com_emb_keys[layer_i-1], part_com_emb_keys[layer_i-1]['dst']], 
                            [last_com_emb_values, last_part_emb_values])
        replace_end = time.perf_counter()
        if num_aggrs_dev[layer_i] < 1:
            if num_aggrs_host[layer_i] >= 1:
                aggrs_cache[layer_i].replace(True)
            else:
                aggrs_cache[layer_i].replace(False,
                            [p_dst, curr_full_keys],
                            [curr_aggr_values, curr_full_aggrs])
        replace_time[1] += time.perf_counter() - replace_end
        replace_time[2] += replace_end - replace_start

        if not curr_com_emb_values.is_cuda:
            torch.cuda.synchronize(device)
        if layer_i < num_model_layers - 1:
            last_com_emb_values = curr_com_emb_values
            last_part_emb_values = part_curr_com_emb
    
    torch.cuda.synchronize(device)
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
    means = np.mean(ls, axis=0)
    print(f'{retrieve_name(ls)[0]}: {means}, {means if isinstance(means, np.float64) else np.sum(means)}') 
    for i in ls:
        print(i, end=',')
    print()
