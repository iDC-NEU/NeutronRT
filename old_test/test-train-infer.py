import argparse
import os
import random

import torch
from torch_geometric.utils import to_torch_coo_tensor
from torch_geometric.datasets import Reddit, Planetoid
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset

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
        num_dim = num_classes
    else:
        num_dim = args.num_hiddens
    emb = torch.ones((data.num_nodes, num_dim), dtype=torch.float32, pin_memory=True)
    embeddings.append(emb)

aggregated_embs = []
for i in range(num_model_layers):
    if i == 0:
        num_dim = num_features
    else:
        num_dim = args.num_hiddens
    emb = torch.ones((data.num_nodes, num_dim), dtype=torch.float32, pin_memory=True)
    aggregated_embs.append(emb)

for layer_i in range(num_model_layers):
    loader = NeighborSampler(data.edge_index,
            node_idx=torch.arange(data.num_nodes),
            sizes=[-1],
            batch_size=batch_size_sample,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    for step, (batch_size, ids, adjs) in enumerate(loader):
        update_x = embeddings[layer_i][ids].to(device)
        sampled_edge_index = adjs[0].to(device)
        update_y, update_aggr = model.layer_wise(layer_i, update_x, sampled_edge_index, batch_size)

        next_nodes = ids[:batch_size]
        embeddings[layer_i+1][next_nodes] = update_y.cpu()
        aggregated_embs[layer_i][next_nodes] = update_aggr.cpu()


embeddings[0] = None
output = {'model': model, 'embeddings': embeddings, 'aggregated_embs': aggregated_embs, 
          'select_index': select_index, 'delete_list': delete_list, 'insert_list': insert_list}

torch.save(output, './data/model_partly_mean.pt')
