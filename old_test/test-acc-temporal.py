import argparse
import itertools
import os
import random
import torch
import time

from model_layer import SAGE,GAT,GCN
from head import head_dict

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler

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
argparser.add_argument('--dataset', type=str, default='uo17')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--test-sizes', type=str, default='-1,-1')
# argparser.add_argument('--sizes', type=str, default='10,10,10')
# argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
# argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
# argparser.add_argument('--buffer-size', type=float, default=1)
# argparser.add_argument('--fallback', type=int, default=0)
argparser.add_argument('--task', type=str, default='link_pred')
args = argparser.parse_args()

train_epoch = 50

path = './data/dataset'
os.makedirs(path, exist_ok=True)
sizes = [int(size) for size in args.sizes.split(',')]

dataset = torch.load('./data/patents_cite/patents_cite_cat.pt')
num_features = dataset['num_features']
num_classes = dataset['num_classes']
year_index = dataset['year_index']
data_x = dataset['x']
data_y = dataset['y']
orig_edge_index = dataset['edge_index']

end_year = len(year_index[0]) - args.num_epochs - 1
i_edge = year_index[2][end_year]
edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(range(i_edge+1)))

data = Data(x=data_x, edge_index=edge_index, y=data_y)

random.seed(0)

start_year = max(0, len(year_index[0]) - end_year - 5)
node_num = min(100000, year_index[1][end_year] - year_index[1][start_year])
sampled_nodes = random.sample(range(year_index[1][start_year], year_index[1][end_year]), node_num)
train_nodes = sampled_nodes[:int(node_num * 0.8)]
train_nodes.sort()
vaild_nodes = sampled_nodes[int(node_num * 0.8):int(node_num * 0.9)]
vaild_nodes.sort()
test_nodes = sampled_nodes[int(node_num * 0.9):]
test_nodes.sort()
split_idx = {'train': torch.tensor(train_nodes), 
             'valid': torch.tensor(vaild_nodes),
             'test': torch.tensor(test_nodes)}

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


def train(model, pred, optimizer, loader):
    model.train()
    pred.train()

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
    return loss, approx_acc

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
pred_list = []
test_res_list = []
test_train_res_list = []

for i in range(args.num_epochs + 1):
    if i != 0:
        i_edge = year_index[2][end_year + i]
        edge_index = torch.index_select(orig_edge_index, 1, torch.tensor(range(i_edge+1)))
        data.edge_index = edge_index

    print(f'Epoch {i}, edge size {data.edge_index.size()} ')

    model = new_model()
    model_list.append(model)
    pred = head_dict[args.task](model.emb, num_classes)
    pred_list.append(pred)

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), 
        lr=args.lr, weight_decay=0.0005)

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

    test_loader = NeighborSampler(data.edge_index,
                                  node_idx=split_idx['test'], 
                                  sizes=[-1] * len(sizes), 
                                  batch_size=args.test_batch_size,
                                  num_workers=args.num_workers)

    # min_num = min(len(split_idx['train']), len(split_idx['test']))
    # test_loader = NeighborSampler(data.edge_index,
    #                               node_idx=split_idx['train'][:min_num], 
    #                               sizes=[-1] * len(sizes), 
    #                               batch_size=args.test_batch_size,
    #                               num_workers=args.num_workers)

    train_times = 0
    for epoch in range(train_epoch):
        train_start = time.perf_counter()
        loss, approx_acc = train(model, pred, optimizer, train_loader)
        train_times += time.perf_counter() - train_start
        val_acc, _ = test(model, val_loader, val_steps=1000)
        if (epoch+1) % 10 == 0:
            print(f'    Epoch: {epoch:02d}, Loss: {loss:.6f}, Approx. Train: {approx_acc:.6f}')
            print(f'    Val Acc: {val_acc:.6f}')
    
    test_start = time.perf_counter()
    test_acc, test_res = test(model, test_loader)
    test_times = time.perf_counter() - test_start
    test_res_list.append(test_res)
    print(f'Test Acc: {test_acc:.6f}, train time {train_times:.6f}, test time {test_times:.6f}')

    if i == args.num_epochs:
        for old_model_i in range(args.num_epochs):
            test_acc, test_res = test(model_list[old_model_i], test_loader)
            print(f'Test Acc in model {old_model_i}: {test_acc:.6f}')
            same_num = 0
            for res_i in range(len(test_res)):
                if test_res_list[old_model_i][res_i] == test_res[res_i]:
                    same_num += 1
            print(f'diff: {len(test_res) - same_num}, same: {same_num}, rate: {(same_num / len(test_res)): .6f}')

