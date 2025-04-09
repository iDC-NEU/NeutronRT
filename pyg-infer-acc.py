import argparse
import os
import random
import torch
import time
import numpy as np

from torch_geometric.nn import GraphSAGE

from task import Task
from lib.dataset import GraphDataset

from torch_geometric.data import Data
from torch_geometric.utils import to_torch_coo_tensor, degree

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=5)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='ogbl-collab')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--compute-type', type=str, default="gpu")
argparser.add_argument('--task', type=str, default='link_pred')
argparser.add_argument('--new-emb', default=False, action='store_true')
args = argparser.parse_args()

save_new_model = False # not args.new_emb
if args.task == 'link_pred':
    args.num_epochs += 1

random.seed(0)
torch.manual_seed(0)

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

dataset = GraphDataset(args.dataset, path, args.batch_size, args.num_epochs)

data = dataset.get_data()
num_features = data['num_features']
num_classes = data['num_classes']
data_num_nodes = data['num_nodes']
data_in_gpu = data['data_in_gpu']
x = data['x'].to(device)
update_lists = data['update_lists']
graph_csr = data['graph_csr']
orignal_edge_index = data['graph_coo']

train_labels = None
test_labels = None
if args.task == 'node':
    train_labels = data['train_labels']
    test_labels = data['update_labels']

dataset = dataset.close()
data = Data(x, edge_index=orignal_edge_index)

def new_model():
    if args.model == 'sage':
        model = GraphSAGE(num_features, args.num_hiddens, num_layers=len(sizes), out_channels=num_classes, dropout=0.5)
    # elif args.model == 'gcn':
    #     model = GCN(num_features, args.num_hiddens, num_classes, num_layers=len(sizes), 
    #                 norm=True)
    # elif args.model == 'gat':
    #     model = GAT(num_features, args.num_hiddens, num_classes, num_layers=len(sizes),
    #                 heads=4)
    else:
        raise NotImplementedError
    return model

downstream_task = Task(args.task, new_model(), num_classes, args.batch_size, args.dataset, device, gpu_infer=args.dataset != 'GDELT')
downstream_task.init_models(data, sizes, save_new_model, train_labels)

if args.new_emb:
    print('old model new emb')
    node_emb = None
else:
    print('old model old emb')
    node_emb2 = downstream_task.inference(data)

# for i in range(len(node_emb[1])):
#     print(torch.allclose(node_emb[1][i], load_emb[1][i], atol=1e-6))

eval_data = []
list_acc = []
list_f1 = []
list_f1_w = []

retrain_times = []
inference_times = []

num_updates = args.num_epochs
if args.task == 'link_pred':
    num_updates -= 1
for i in range(num_updates):
    insert_edges = update_lists[i][0]
    delete_edges = update_lists[i][1]
    edge_index = data.edge_index
    edge_index = torch.cat([edge_index, insert_edges], dim=1)
    edge_index = to_torch_coo_tensor(edge_index).indices().to(torch.int64)
    data.edge_index = edge_index

    retrain_start = time.perf_counter()
    # downstream_task.init_models(data, sizes, True, train_labels)
    retrain_end = time.perf_counter()
    retrain_times.append(retrain_end - retrain_start)

    infer_start = time.perf_counter()
    node_emb = downstream_task.inference(data)
    infer_end = time.perf_counter()
    inference_times.append(infer_end - infer_start)

    print(f'Epoch {i}, edge size {data.edge_index.size()} ')
    if args.task == 'link_pred':
        output = downstream_task.test(data, update_lists[i+1], node_emb)
        list_acc.append(output[0])
        eval_data.append((output[1], output[2]))
    else:
        output = downstream_task.test(data, test_labels[i], node_emb)
        acc, f1, f1_w = output
        list_acc.append(acc)
        list_f1.append(f1)
        list_f1_w.append(f1_w)

if args.task == 'link_pred':
    print(retrain_times)
    print(inference_times)
    print([t1 + t2 for t1, t2 in zip(retrain_times, inference_times)])
    print(list_acc)
    print(f'Average AUC: {np.mean(list_acc)}')
    # if save_new_model:
    # torch.save(eval_data, f'/home/jiangqsh/nus/roland/eval_edge_data_{args.dataset}.pt')
else:
    print(list_acc)
    print(f'Average ACC: {np.mean(list_acc)}, Average F1: {np.mean(list_f1)}, Average F1_w: {np.mean(list_f1_w)}')
