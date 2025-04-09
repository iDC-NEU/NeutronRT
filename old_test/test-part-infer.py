import argparse
import os
import random
import time
import numpy as np

import torch
from torch_geometric.utils import to_torch_coo_tensor, degree, spmm
from torch_geometric.datasets import Reddit, Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

from lib.pma_graph import PMAGraph
from lib.gpu_cache import GPUCache, check_part_compute_state

from model_layer import SAGE,GAT,GCN

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=6)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--emb-host-ratio', type=str, default='1,0.1,1')
argparser.add_argument('--emb-cache-ratio', type=str, default='0.1,0.1,0')
argparser.add_argument('--aggr-host-ratio', type=str, default='1,1')
argparser.add_argument('--aggr-cache-ratio', type=str, default='0.1,0.1')
argparser.add_argument('--compute-type', type=str, default="gpu")
args = argparser.parse_args()

sizes = [int(size) for size in args.sizes.split(',')]

if args.compute_type == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:%d' % args.gpu)
    torch.cuda.set_device(device)

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
x = torch.ones(data.x.size(), dtype=torch.float32, pin_memory=True)
x.copy_(data.x)
y = data.y.view(-1).to(torch.int64)
data.x = x
data.y = y

input = torch.load('./data/model_partly_mean.pt')

# Define model
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

# model = new_model()
model = input['model']
model.eval()

# Define edge update
# delete_list = []
# insert_list = []
# select_index = set(range(orig_edge_index.size()[-1]))

# for i in range(args.num_epochs):
#     random.seed(i)
#     index = random.sample(select_index, args.batch_size)
#     pos = len(index) // 2
#     delete_list.append(set(index[:pos]))
#     insert_list.append(set(index[pos:]))
#     select_index -= insert_list[i]

# delete_list = delete_list[::-1]
# insert_list = insert_list[::-1]

select_index = input['select_index']
delete_list = input['delete_list']
insert_list = input['insert_list']

select_index_list = list(select_index)
select_index_list.sort()
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
data.edge_index = edge_index

data_degree = degree(data.edge_index[0], data.num_nodes)
_, topk_degree = torch.sort(data_degree, descending=True)

# Define GPU cache 
data_cache = [] # embedding cache
# storing all embeddings of x0 and x_n on host for n-layer model
# num_item_host = [1, 1, 1]
num_item_host = [float(ra) for ra in args.emb_host_ratio.split(',')]
# num_item_dev = [5, 10, 0]
# num_item_dev = [0.1, 0.1, 0] # todo: capacity
num_item_dev = [float(ra) for ra in args.emb_cache_ratio.split(',')]
for i in range(num_model_layers + 1):
    # pre = num_item_dev[i] * 1024 * 1024 * 1024 // (x.element_size() * num_features)
    # num_item_dev[i] = min(pre, data.num_nodes) 
    num_item_dev[i] = int(num_item_dev[i] * data.num_nodes)
    if i > 0 and i < num_model_layers:
        num_item_host[i] = int(num_item_host[i] * data.num_nodes)

for i in range(num_model_layers + 1):
    num_dev = num_item_dev[i]
    if  i == 0:
        num_dim = num_features
        num_host = data.num_nodes
        host_key = torch.arange(data.num_nodes, device=device)
        dev_key, _ = torch.sort(topk_degree[:num_dev])
        emb = x
        cache_emb = x[dev_key].to(device)
    elif i == num_model_layers:
        num_dim = num_classes
        num_host = data.num_nodes
        host_key = torch.arange(data.num_nodes, device=device)
        dev_key = torch.empty((0), dtype=torch.int64, device=device)
        emb = input['embeddings'][i].pin_memory()
        cache_emb = input['embeddings'][i][dev_key.cpu()].to(device)
        # emb = torch.ones((num_host, num_dim), dtype=torch.float32, pin_memory=True)
        # cache_emb = torch.empty((num_dev, num_dim), dtype=torch.float32, device=device)
    else:
        num_dim = args.num_hiddens
        num_host = num_item_host[i]
        if num_dev + num_host > data.num_nodes:
            num_host = data.num_nodes - num_dev
        dev_key, _ = torch.sort(topk_degree[:num_dev])
        host_key, _ = torch.sort(topk_degree[num_dev:num_dev+num_host])
        emb = input['embeddings'][i][host_key.cpu()].pin_memory()
        cache_emb = input['embeddings'][i][dev_key.cpu()].to(device)
        # emb = torch.ones((num_host, num_dim), dtype=torch.float32, pin_memory=True)
        # cache_emb = torch.ones((num_dev, num_dim), dtype=torch.float32, device=device)
    data_cache.append(GPUCache(data.num_nodes, host_key, emb, 
                                dev_key, cache_emb, device))

test_emb = [x, input['embeddings'][1], input['embeddings'][2]]

aggrs_cache = [] # aggregated embedding cache
# num_aggrs_host = [0, 0]
num_aggrs_host = [float(ra) for ra in args.aggr_host_ratio.split(',')]
# num_aggrs_dev = [0, 0]
num_aggrs_dev = [float(ra) for ra in args.aggr_cache_ratio.split(',')]
for i in range(num_model_layers):
    num_aggrs_host[i] = int(num_aggrs_host[i] * data.num_nodes)
    num_aggrs_dev[i] = int(num_aggrs_dev[i] * data.num_nodes)

for i in range(num_model_layers):
    if i == 0:
        num_dim = num_features
    else:
        num_dim = args.num_hiddens
    num_dev = num_aggrs_dev[i]
    num_host = num_aggrs_host[i]
    if num_dev + num_host > data.num_nodes:
        num_host = data.num_nodes - num_dev
    dev_key, _ = torch.sort(topk_degree[:num_dev])
    host_key, _ = torch.sort(topk_degree[num_dev:num_dev+num_host])
    emb = input['aggregated_embs'][i][host_key.cpu()].pin_memory()
    cache_emb = input['aggregated_embs'][i][dev_key.cpu()].to(device)
    # emb = torch.ones((num_host, num_dim), dtype=torch.float32, pin_memory=True)
    # cache_emb = torch.ones((num_dev, num_dim), dtype=torch.float32, device=device)
    aggrs_cache.append(GPUCache(data.num_nodes, host_key, emb,
                                 dev_key, cache_emb, device))


def error(a=None, b=None):
    if a is not None:
        size = torch.where(torch.all(torch.isclose(a, b, atol=1e-3), dim=1) == False)[0].size()
        print(size)
    print('err')

# Define PMA graph
graph = PMAGraph(data.num_nodes, data.edge_index, device)

update_graph_times = []
inference_times = []
sample_times = []
transfer_times = []
compute_times = []
sparse_times = []

for i in range(args.num_epochs):
    insert_edges = list(insert_list[i])
    insert_edges.sort()
    insert_edges = torch.index_select(orig_edge_index, 1, torch.tensor(insert_edges))
    delete_edges = list(delete_list[i])
    delete_edges.sort()
    delete_edges = torch.index_select(orig_edge_index, 1, torch.tensor(delete_edges))

    # todo: 1. test only insert 2. test need to compute embedding

    select_index |= insert_list[i]
    select_index -= delete_list[i]
    select_index_list = list(select_index)
    select_index_list.sort()
    edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
    data.edge_index = edge_index

    # ahead, part, full
    sample_time = [0, 0, 0]
    compute_time = [0, 0, 0]
    transfer_time = [0, 0, 0]
    sparse_time = 0

    update_start = time.perf_counter()
    last_in_degree = graph.clone_degree()
    graph.insert_edges(insert_edges)
    graph.delete_edges(delete_edges)

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
        src, mapped_dst, o_type, dst_indices = graph.next_affact_nodes(insert_edges, delete_edges, dst_last_layer)
        
        # check both dst and src at the same time
        # return bool for dst and src (src and type)
        can_part, need_src = check_part_compute_state(aggrs_cache[layer_i], data_cache[layer_i], mapped_dst, src, dst_indices)
        p_src = src[need_src]
        p_type = o_type[need_src]
        _, p_extend, p_counts = torch.unique_consecutive(p_src, return_inverse=True, return_counts=True)
        p_indices = torch.zeros(p_counts.size(0) + 1, dtype=torch.int64, device=device)
        torch.cumsum(p_counts, 0, out=p_indices[1:])
        if layer_i == 0:
            old_key = None
        else:
            old_key = torch.unique(p_src[p_type <= 0])
        part_com_emb_keys.append({'dst': mapped_dst[can_part], 
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
            part_dst = part_com_emb_keys[layer_i]['dst']
            part_dst_type = data_cache[layer_i].check_state(part_dst)
            full_com_emb_keys[layer_i-1] = torch.unique(torch.cat([full_com_emb_keys[layer_i-1], 
                src[src_type < 0], dst[dst_type < 0], part_dst[part_dst_type < 0]]))

        relations[layer_i] = (src, dst_indices, dst_contains)

    sample_time[0] += time.perf_counter() - sample_start

    last_com_emb_values = None
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
            affect_new_x, test_miss = data_cache[layer_i].query(new_key)

        if torch.any(test_miss):
            error()

        if layer_i + 1 < num_model_layers:
            next_layer_i = layer_i + 1
            next_old_key = part_com_emb_keys[next_layer_i]['old']
            next_affect_old_x, test_miss = data_cache[next_layer_i].query(next_old_key, 0)
            next_old_x = next_affect_old_x

        if torch.any(test_miss):
            error()

        transfer_end = time.perf_counter()
        sample_time[1] += sample_end - sample_start
        transfer_time[1] += transfer_end - sample_end

        # if layer_i > 0:
        #     batch_size_sample = 1
        # update partly computed aggreated embedding
        next_start = 0
        curr_aggr_values = torch.zeros((p_dst.size(0), aggrs_cache[layer_i].num_feat), 
                    dtype=torch.float32, device=device)
        # unchanged = torch.zeros_like(dst_nodes, dtype=torch.bool, device=device)
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
            tmp_old = selected_src[old_select_index]
            tmp_new = selected_src[new_select_index]
            old_mapped_ids = torch.empty((2, tmp_old.size(0)), dtype=torch.int64, device=device)
            new_mapped_ids = torch.empty((2, tmp_new.size(0)), dtype=torch.int64, device=device)
            torch.searchsorted(old_key, tmp_old, out=old_mapped_ids[1])
            torch.searchsorted(new_key, tmp_new, out=new_mapped_ids[1])
            
            sample_end = time.perf_counter()
            
            # old aggreated embedding
            old_ax, test_miss = aggrs_cache[layer_i].query(next_nodes)
            if torch.any(test_miss):
                error()

            # new_x_target, test_miss = data_cache[layer_i].query(next_nodes)
            # if torch.any(test_miss):
            #     error()
            
            transfer_end = time.perf_counter()

            compute_type = model.get_aggr(layer_i)
            if compute_type == 'mean' or compute_type == 'sum':
                nodes_coo_src = p_extend[dst_indices[next_start]:dst_indices[next_end]] - next_start
                torch.masked_select(nodes_coo_src, old_select_index, out=old_mapped_ids[0])
                old_ids_coo = torch.sparse_coo_tensor(old_mapped_ids, 
                                                  torch.ones_like(old_mapped_ids[0], dtype=torch.float32), 
                                                  (next_end - next_start, old_key.size(0)))
                torch.masked_select(nodes_coo_src, new_select_index, out=new_mapped_ids[0])
                new_ids_coo = torch.sparse_coo_tensor(new_mapped_ids, 
                                                  torch.ones_like(new_mapped_ids[0], dtype=torch.float32), 
                                                  (next_end - next_start, new_key.size(0)))
                sparse_end = time.perf_counter()
                # new_ax = torch.zeros_like(old_ax, device=device, dtype=torch.float32)
                torch.sub(torch.sparse.mm(new_ids_coo, affect_new_x), 
                          torch.sparse.mm(old_ids_coo, affect_old_x), 
                          out=curr_aggr_values[next_start:next_end])
                if compute_type == 'mean':
                    sum_ax = torch.mul(old_ax, last_in_degree[next_nodes].view(-1, 1))
                    torch.add(curr_aggr_values[next_start:next_end], sum_ax,   
                              out=curr_aggr_values[next_start:next_end])
                    torch.div(curr_aggr_values[next_start:next_end], 
                              graph.in_degree[next_nodes].view(-1, 1), 
                              out=curr_aggr_values[next_start:next_end])
                else:
                    torch.add(curr_aggr_values[next_start:next_end], old_ax, 
                              out=curr_aggr_values[next_start:next_end])
            else:
                raise NotImplementedError

            # update_y = model.layer_wise_partly(layer_i, new_ax, new_x_target)
            compute_end = time.perf_counter()
            """
            from torch_geometric.loader import NeighborSampler
            loader = NeighborSampler(data.edge_index, node_idx=next_nodes,
                        sizes=[-1], batch_size=batch_size_sample,
                        shuffle=False, num_workers=args.num_workers)
            for step, (batch_size, ids, adjs) in enumerate(loader):
                update_x2, _ = data_cache[layer_i].query(ids.to(device), 0)
                sampled_edge_index = adjs[0].to(device)
                update_y2, update_aggr2 = model.layer_wise(layer_i, update_x2, sampled_edge_index, batch_size)
                test_agg = curr_aggr_values[next_start:next_end]
                err = torch.allclose(update_aggr2, test_agg, atol=1e-3)
                if not err:
                    num_false = torch.where(torch.all(torch.isclose(update_aggr2, test_agg, atol=1e-3), dim=1) == False)[0].size()
                    print(num_false)
                    error()
            """
            # torch.all(torch.isclose(new_ax, old_ax), dim=1, out=unchanged[next_start:next_end])
            aggrs_cache[layer_i].update(next_nodes, 
                        curr_aggr_values[next_start:next_end]) # may async
            # data_cache[layer_i+1].update(next_nodes, update_y) # may async
            next_start = next_end

            sample_time[1] += sample_end - sample_start
            transfer_time[1] += transfer_end - sample_end
            sparse_time += sparse_end - transfer_end
            compute_time[1] += compute_end - sparse_end

        # dst_last_layer = dst_nodes[unchanged == False]

        # update full
        curr_full_keys = full_com_emb_keys[layer_i]
        if layer_i < num_model_layers - 1:
            curr_com_emb_values = torch.zeros((curr_full_keys.size(0), data_cache[layer_i+1].num_feat), 
                        dtype=torch.float32, device=device)
            
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

            csr = torch.sparse_csr_tensor(dst_idx, src_idx, 
                torch.ones_like(src_idx, dtype=torch.float32), 
                size=(next_end - next_start, src_emb.size(0)),
                device=device)
            aggr_out = spmm(csr, src_emb, model.get_aggr(layer_i))

            _, miss = aggrs_cache[layer_i].query(next_nodes, out=aggr_out, ignore=False)

            # if not torch.equal(miss, dst_contains[next_start:next_end]  < 0 ):
            #     error()

            update_y = model.layer_wise_partly(layer_i, aggr_out, new_x_target)
            compute_end = time.perf_counter()

            aggrs_cache[layer_i].update(next_nodes, aggr_out, miss == False) 
            data_cache[layer_i+1].update(next_nodes, update_y) # may async
            if layer_i < num_model_layers - 1:
                if curr_com_emb_values.is_cuda:
                    curr_com_emb_values[next_start:next_end] = update_y
                else:
                    curr_com_emb_values[next_start:next_end] = update_y.to(device='cpu', non_blocking=True)

            next_start = next_end

            sample_time[2] += sample_end - sample_start
            transfer_time[2] += transfer_end - sample_end
            compute_time[2] += compute_end - transfer_end
        
        # update partly computed embedding
        while next_start < p_dst.size(0):
            transfer_start = time.perf_counter()
            next_end = min(next_start + batch_size_sample, p_dst.size(0))
            next_nodes = p_dst[next_start:next_end]

            new_x_target, target_miss = data_cache[layer_i].query(next_nodes)
            if layer_i > 0 and torch.any(target_miss):
                target_index = torch.searchsorted(full_com_emb_keys[layer_i-1], next_nodes[target_miss])
                if last_com_emb_values.is_cuda:
                    new_x_target[target_miss] = last_com_emb_values[target_index]
                else:
                    new_x_target[target_miss] = last_com_emb_values[target_index.cpu()].to(device)
                # if not torch.equal(next_nodes[target_miss], full_com_emb_keys[layer_i-1][target_index]):
                #     error()

            transfer_end = time.perf_counter()
            update_y = model.layer_wise_partly(layer_i, 
                                               curr_aggr_values[next_start:next_end], 
                                               new_x_target)
            compute_end = time.perf_counter()

            from torch_geometric.loader import NeighborSampler
            loader = NeighborSampler(data.edge_index, node_idx=next_nodes,
                        sizes=[-1], batch_size=batch_size_sample,
                        shuffle=False, num_workers=args.num_workers)
            for step, (batch_size, ids, adjs) in enumerate(loader):
                # update_x2 = data_cache[layer_i].host_emb[ids].to(device)
                update_x2 = test_emb[layer_i][ids].to(device)
                sampled_edge_index = adjs[0].to(device)
                update_y2, update_aggr2 = model.layer_wise(layer_i, update_x2, sampled_edge_index, batch_size)
                test_emb[layer_i+1][ids[:batch_size]] = update_y2.cpu()

                err = torch.allclose(new_x_target, update_x2[:batch_size], atol=1e-3)
                if not err:
                    error(new_x_target, update_x2[:batch_size])
                test_agg = curr_aggr_values[next_start:next_end]
                err = torch.allclose(update_aggr2, test_agg, atol=1e-3)
                if not err:
                    error(update_aggr2, test_agg)
                err = torch.allclose(update_y2, update_y, atol=1e-3)
                if not err:
                    error(update_y2, update_y)

            transfer_time[1] += transfer_end - transfer_start
            compute_time[1] += compute_end - transfer_end
            data_cache[layer_i+1].update(next_nodes, update_y) # may async
            next_start = next_end

        if not curr_com_emb_values.is_cuda:
            torch.cuda.synchronize(device)
        if layer_i < num_model_layers - 1:
            last_com_emb_values = curr_com_emb_values
        # may async
        # if num_item_dev[layer_i] <= data.num_nodes:
        #     data_cache[layer_i].replace()
    
    infer_end = time.perf_counter()
    inference_times.append(infer_end - infer_start)
    sample_times.append(sample_time)
    transfer_times.append(transfer_time)
    compute_times.append(compute_time)
    sparse_times.append(sparse_time)

    # nodes_recompute = dst_last_layer
    # batch_size_sample = 1
    # from torch_geometric.loader import NeighborSampler
    # loader = NeighborSampler(data.edge_index, node_idx=nodes_recompute,
    #             sizes=[-1, -1], batch_size=batch_size_sample,
    #             shuffle=False, num_workers=args.num_workers)
    # last_start = 0
    # for step, (batch_size, ids, adjs) in enumerate(loader):
    #     nodes = nodes_recompute[last_start:last_start+batch_size]
    #     adjs = [adj.to(device) for adj in adjs]
    #     batch_input = data.x[ids].to(device)
    #     out1, aggr1 = model(batch_input, adjs)
    #     last_start += batch_size

    #     output2, test_miss = data_cache[-1].query(nodes)
    #     if torch.any(test_miss):
    #         error()

    #     aggr2, test_miss = aggrs_cache[-1].query(nodes)

    #     err = torch.allclose(aggr1[test_miss==False], aggr2[test_miss==False], atol=1e-3)
    #     if not err:
    #         error()

    #     err = torch.allclose(out1, output2, atol=1e-3)
    #     if not err:
    #         error()


import inspect

def retrieve_name(var):
    '''
    utils:
    get back the name of variables
    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


for ls in [update_graph_times, inference_times, 
           sample_times, transfer_times, 
           compute_times, sparse_times]:
    print(f'{retrieve_name(ls)[0]}: {np.mean(ls, axis=0)}')
    for i in ls:
        print(i, end=',')
    print()
