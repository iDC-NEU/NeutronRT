import argparse
import os
import random
import torch
import time

from model import SAGE,GAT,GCN

import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=4)
argparser.add_argument('--num-epochs', type=int, default=10)
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

train_epoch = 1

path = './data/dataset'
os.makedirs(path, exist_ok=True)

if args.dataset == 'reddit':
    dataset = Reddit(path)
    data = dataset[0]
    split_idx = {'train': data.train_mask.nonzero(as_tuple=False).view(-1), 
                 'valid': data.val_mask.nonzero(as_tuple=False).view(-1), 
                 'test': data.test_mask.nonzero(as_tuple=False).view(-1)}
else:
    dataset = PygNodePropPredDataset(args.dataset, path)
    split_idx = dataset.get_idx_split()

num_features = dataset.num_features
num_classes = dataset.num_classes

sizes = [int(size) for size in args.sizes.split(',')]

data = dataset[0]
orig_edge_index = data.edge_index

select_index = range(orig_edge_index.size()[-1])
insert_list = random.sample(select_index, orig_edge_index.size()[-1] // 2)
select_index = set(select_index) - set(insert_list)

select_index_list = list(select_index)
select_index_list.sort()
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
data.edge_index = edge_index

x = data.x
y = data.y.view(-1).to(torch.long)

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


def train(model, optimizer, loader):
    model.train()

    total_loss = total_correct = total_examples = 0

    for i, (batch_size, ids, adjs) in enumerate(loader):
        # start_time = time.perf_counter()
        adjs = [adj.to(device) for adj in adjs]
        batch_inputs = x[ids].to(device)
        batch_labels = y[ids[:batch_size]].to(device)

        optimizer.zero_grad()
        out = model(batch_inputs, adjs)
        loss = F.cross_entropy(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels).sum())
        total_examples += batch_size

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / total_examples
    print(f'Epoch: {epoch:02d}, Loss: {loss:.6f}, Approx. Train: {approx_acc:.6f}')
    
@torch.no_grad()
def test(model, loader, val_steps = None):
    model.eval()

    total_correct = total_examples = 0
    test_results = []

    for i, (batch_size, ids, adjs) in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break

        adjs = [adj.to(device) for adj in adjs]
        batch_input = x[ids].to(device)
        batch_labels = y[ids[:batch_size]].to(device)

        out = model(batch_input, adjs)
        pred = out.argmax(dim=-1)
        
        total_correct += int(pred.eq(batch_labels).sum())
        total_examples += batch_size

        if val_steps is None:
            test_results.extend(pred.tolist())

    return total_correct / total_examples, test_results


model_list = []
test_res_list = []
test_train_res_list = []

for i in range(args.num_epochs + 1):
    if i != 0:
        insert_len = len(insert_list) // args.num_epochs
        start_insert = (i - 1) * insert_len
        end_insert = i * insert_len if i != args.num_epochs else len(insert_list)
        select_index |= set(insert_list[start_insert:end_insert])
        select_index_list = list(select_index)
        select_index_list.sort()
        edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(select_index_list))
        data.edge_index = edge_index

    print(f'Epoch {i}, edge size {data.edge_index.size()} ')

    model = new_model()
    model_list.append(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    train_loader = NeighborSampler(data.edge_index,
                                   node_idx=split_idx['train'], 
                                   sizes=sizes, 
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    
    val_loader = NeighborSampler(data.edge_index,
                                 node_idx=split_idx['valid'], 
                                 sizes=sizes, 
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    # test_loader = NeighborSampler(data.edge_index,
    #                               node_idx=split_idx['test'], 
    #                               sizes=[-1] * len(sizes), 
    #                               batch_size=args.test_batch_size,
    #                               num_workers=args.num_workers)

    min_num = min(len(split_idx['train']), len(split_idx['test']))
    test_loader = NeighborSampler(data.edge_index,
                                  node_idx=split_idx['train'][:min_num], 
                                  sizes=[-1] * len(sizes), 
                                  batch_size=args.test_batch_size,
                                  num_workers=args.num_workers)

    for epoch in range(train_epoch):
        train(model, optimizer, train_loader)
        # val_acc, _ = test(model, val_loader, val_steps=1000)
        # print(f'Val Acc: {val_acc:.6f}')
    
    test_acc, test_res = test(model, test_loader)
    test_res_list.append(test_res)
    print(f'Test Acc: {test_acc:.6f}')

    if i == args.num_epochs:
        for old_model_i in range(args.num_epochs):
            test_acc, test_res = test(model_list[old_model_i], test_loader)
            print(f'Test Acc in model {old_model_i}: {test_acc:.6f}')
            same_num = 0
            for res_i in range(len(test_res)):
                if test_res_list[old_model_i][res_i] == test_res[res_i]:
                    same_num += 1
            print(f'diff: {len(test_res) - same_num}, same: {same_num}, rate: {(same_num / len(test_res)): .6f}')

