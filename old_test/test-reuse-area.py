import argparse
import os
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, add_random_edge
from model import SAGE,GAT,GCN
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_torch_csr_tensor, to_torch_csc_tensor, degree
from ogb.nodeproppred import PygNodePropPredDataset


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=5)
argparser.add_argument('--batch-size', type=int, default=2)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='reddit')
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

# print(args.dataset, data.is_directed())
# out_degree = degree(data.edge_index[0])
# in_degree = degree(data.edge_index[1])
# print("edge num:", len(data.edge_index[0]), "node num:", len(out_degree), len(in_degree))
# print("out_degree:", out_degree.max(), out_degree.min(), out_degree.mean())
# print("in_degree:", in_degree.max(), in_degree.min(), in_degree.mean())


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

for i in range(args.num_epochs):
    select_index |= insert_list[i]
    select_index -= delete_list[i]
    select_index_list = list(select_index)
    select_index_list.sort()
    edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
    data = Data(x=x, edge_index=edge_index)

    update_index = list(insert_list[i] | delete_list[i])
    update_edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(update_index))
    update_nodes = set()
    # affect_area = set()

    for edge_i in range(len(update_edge_index[1])):
       update_nodes.add(int(update_edge_index[1][edge_i])) # dst
    #    affect_area.add((int(update_edge_index[0][edge_i]), int(update_edge_index[1][edge_i])))
    
    recompute_nodes = update_nodes.copy()

    graph_csr = to_torch_csr_tensor(data.edge_index)
    for j in range(1, num_model_layers):
        next_layer_update_nodes = set()
        for src in update_nodes:
            for edge_index in range(graph_csr.crow_indices()[src], graph_csr.crow_indices()[src+1]):
                dst = int(graph_csr.col_indices()[edge_index])
                next_layer_update_nodes.add(dst)
        update_nodes = next_layer_update_nodes
        recompute_nodes |= update_nodes

    changed_vertices = recompute_nodes.copy()

    affect_all_nodes = recompute_nodes.copy()
    all_edges_num = 0
    recalculated_edges_num = 0

    graph_csc = to_torch_csc_tensor(data.edge_index)
    for j in range(num_model_layers):
        next_layer_nodes = set()
        for dst in recompute_nodes:
            current_edges_num = int(graph_csc.ccol_indices()[dst+1]) - int(graph_csc.ccol_indices()[dst])
            all_edges_num += current_edges_num
            if dst in changed_vertices:
                recalculated_edges_num += current_edges_num
            # else:
            #     print(dst, "not in changed_vertices")
            for edge_index in range(graph_csc.ccol_indices()[dst], graph_csc.ccol_indices()[dst+1]):
                src = int(graph_csc.row_indices()[edge_index])
                if src not in affect_all_nodes:
                    next_layer_nodes.add(src)
        recompute_nodes = next_layer_nodes
        affect_all_nodes |= recompute_nodes


    # affect_area = list(affect_area)
    # affect_area.sort()
    # edges_num = len(affect_area)
    
    # affect_edge_index = [[0] * edges_num, [0] * edges_num]
    # affect_all_nodes = set()
    # for step, edge in enumerate(affect_area):
    #     affect_edge_index[0][step] = edge[0]
    #     affect_edge_index[1][step] = edge[1]
    #     affect_all_nodes.add(edge[0])
    #     affect_all_nodes.add(edge[1])
    
    # affect_data = Data(x=x, edge_index=torch.tensor(affect_edge_index))

    # affect_csc = affect_data.csc()
    # recalculated_edges_num = 0
    # for node in changed_vertices:
    #     recalculated_edges_num += affect_csc[0][node+1] - affect_csc[0][node]

    print("Epoch:", i, 
          ", nodes num:", len(affect_all_nodes), 
          ", edges num:", all_edges_num, 
          ", changed vertices num:", len(changed_vertices),
          ", recalculated edges num:", recalculated_edges_num, 
          ", recalculated rate:", recalculated_edges_num / all_edges_num)
