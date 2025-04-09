import argparse
import os
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, add_random_edge
from model import SAGE,GAT,GCN
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=5)
argparser.add_argument('--batch-size', type=int, default=10)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbn-products')
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
argparser.add_argument('--buffer-size', type=float, default=1)
argparser.add_argument('--fallback', type=int, default=0)
args = argparser.parse_args()


path = './data/dataset'
os.makedirs(path, exist_ok=True)
num_model_layers = 2
batch_size_sample = 2000

if args.dataset == 'reddit':
    data = Reddit(path)[0]
else:
    data = PygNodePropPredDataset(args.dataset, path)[0]

orig_edge_index = data.edge_index
# x = data.x
x = None
y = data.y

delete_list = []
insert_list = []
remove_ahead = set()
select_index = range(orig_edge_index.size()[-1])

for i in range(args.num_epochs):
    index = random.sample(select_index, args.batch_size)
    delete_list.append(set(index[:len(index) // 2]))
    index_set = set(index[len(index) // 2:])
    insert_list.append(index_set)
    remove_ahead |= index_set

select_index = set(select_index) - remove_ahead
select_index_list = list(select_index)
select_index_list.sort()
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))

# last_changed_node_list = [set() for i in range(num_model_layers)]
last_unchanged_node_emb_list = [set() for i in range(num_model_layers)]

for i in range(args.num_epochs):
    select_index |= insert_list[i]
    select_index -= delete_list[i]
    select_index_list = list(select_index)
    select_index_list.sort()
    edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
    data = Data(x=x, edge_index=edge_index)

    update_index = list(insert_list[i] | delete_list[i])
    update_edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(update_index))
    update_nodes = [set() for i in range(num_model_layers)]

    for edge in update_edge_index[1]:
       update_nodes[0].add(int(edge)) # dst

    graph_csr = data.csr()
    for j in range(1, num_model_layers):
        for node in update_nodes[j-1]:
            for edge_index in range(graph_csr[0][node], graph_csr[0][node+1]):
                edge = int(graph_csr[1][edge_index])
                update_nodes[j].add(edge)

    current_unchanged_node_emb_list = [set() for i in range(num_model_layers)]
    recompute_num = [0] * num_model_layers

    recompute_nodes = set()
    for layer_i in range(num_model_layers):
        recompute_nodes |= update_nodes[layer_i]
        recompute_num[layer_i] = len(recompute_nodes)
        loader = NeighborSampler(data.edge_index,
            node_idx=torch.tensor(list(recompute_nodes)),
            sizes=[-1],
            batch_size=batch_size_sample,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for step, (batch_size, ids, adjs) in enumerate(loader):
            for node_i in adjs.edge_index[0]:
                node = int(ids[node_i])
                if layer_i == 0:
                    current_unchanged_node_emb_list[layer_i].add(node)
                else:
                    need_recompute = False
                    for layer_j in range(layer_i):
                        if node in update_nodes[layer_j]:
                            need_recompute = True
                            break
                    if not need_recompute:
                        current_unchanged_node_emb_list[layer_i].add(node)

    print("Epoch:", i, 
          ", updated nodes num:", len(recompute_nodes))
    for j in range(num_model_layers):
        print("Layer:", j, 
              ", recompute nodes:", recompute_num[j],
              ", unchanged nodes:", len(current_unchanged_node_emb_list[j]),
              ", overlap nodes:", len(current_unchanged_node_emb_list[j] & last_unchanged_node_emb_list[j]), 
              ", loaded nodes:", len(current_unchanged_node_emb_list[j] - last_unchanged_node_emb_list[j]))

    last_unchanged_node_emb_list = current_unchanged_node_emb_list
