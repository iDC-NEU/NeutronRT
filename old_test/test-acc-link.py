import argparse
import os
import random
import torch
import time
from tqdm import tqdm

import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import GAE, RGCNConv



# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=5)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1024)
argparser.add_argument('--test-batch-size', type=int, default=1024)
argparser.add_argument('--num-workers', type=int, default=10)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--model', type=str, default="sage")
argparser.add_argument('--dataset', type=str, default='FB15k-237')
argparser.add_argument('--sizes', type=str, default='25,10')
argparser.add_argument('--test-sizes', type=str, default='-1,-1')
# argparser.add_argument('--sizes', type=str, default='10,10,10')
# argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
# argparser.add_argument('--features', type=int, default=128)
argparser.add_argument('--compute-type', type=str, default="gpu")
# argparser.add_argument('--buffer-size', type=float, default=1)
# argparser.add_argument('--fallback', type=int, default=0)
args = argparser.parse_args()

train_epoch = 100

if args.compute_type == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:%d' % args.gpu)
    torch.cuda.set_device(device)


path = './data/dataset'
path = os.path.join(path, 'RLPD')
os.makedirs(path, exist_ok=True)

dataset = RelLinkPredDataset(path, args.dataset)
data = dataset[0].to(device)

orig_edge_index = data.edge_index
orig_edge_type = data.edge_type

select_index = range(orig_edge_index.size()[-1])
insert_list = random.sample(select_index, orig_edge_index.size()[-1] // 2)
select_index = set(select_index) - set(insert_list)

select_index_list = list(select_index)
select_index_list.sort()
select_index_list = torch.tensor(select_index_list, device=device)
edge_index = torch.index_select(orig_edge_index, 1, select_index_list)
edge_type = torch.index_select(orig_edge_type, 0, select_index_list)
data.edge_index = edge_index
data.edge_type = edge_type


# Define model
class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ),
                                              device=neg_edge_index.device)
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ),
                                              device=neg_edge_index.device)
    return neg_edge_index


def train(model, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.edge_index, data.edge_type)

    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.train_edge_index, data.num_nodes)
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)

@torch.no_grad()
def test(model):
    model.eval()
    z = model.encode(data.edge_index, data.edge_type)

    valid_mrr = compute_mrr(model, z, data.valid_edge_index, data.valid_edge_type)
    test_mrr = compute_mrr(model, z, data.test_edge_index, data.test_edge_type)

    return valid_mrr, test_mrr

@torch.no_grad()
def compute_rank(ranks):
    # fair ranking prediction as the average
    # of optimistic and pessimistic ranking
    true = ranks[0]
    optimistic = (ranks > true).sum() + 1
    pessimistic = (ranks >= true).sum()
    return (optimistic + pessimistic).float() * 0.5


@torch.no_grad()
def compute_mrr(model, z, edge_index, edge_type):
    ranks = []
    for i in tqdm(range(edge_type.numel())):
        (src, dst), rel = edge_index[:, i], edge_type[i]

        # Try all nodes as tails, but delete true triplets:
        tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            tail_mask[tails[(heads == src) & (types == rel)]] = False

        tail = torch.arange(data.num_nodes)[tail_mask]
        tail = torch.cat([torch.tensor([dst]), tail])
        head = torch.full_like(tail, fill_value=src)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(tail, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

        # Try all nodes as heads, but delete true triplets:
        head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        for (heads, tails), types in [
            (data.train_edge_index, data.train_edge_type),
            (data.valid_edge_index, data.valid_edge_type),
            (data.test_edge_index, data.test_edge_type),
        ]:
            head_mask[heads[(tails == dst) & (types == rel)]] = False

        head = torch.arange(data.num_nodes)[head_mask]
        head = torch.cat([torch.tensor([src]), head])
        tail = torch.full_like(head, fill_value=dst)
        eval_edge_index = torch.stack([head, tail], dim=0)
        eval_edge_type = torch.full_like(head, fill_value=rel)

        out = model.decode(z, eval_edge_index, eval_edge_type)
        rank = compute_rank(out)
        ranks.append(rank)

    return (1. / torch.tensor(ranks, dtype=torch.float)).mean()

model_list = []
test_train_res_list = []

for i in range(args.num_epochs + 1):
    if i != 0:
        insert_len = len(insert_list) // args.num_epochs
        start_insert = (i - 1) * insert_len
        end_insert = i * insert_len if i != args.num_epochs else len(insert_list)
        select_index |= set(insert_list[start_insert:end_insert])
        select_index_list = list(select_index)
        select_index_list.sort()
        select_index_list = torch.tensor(select_index_list, device=device)
        edge_index = torch.index_select(orig_edge_index, 1, select_index_list)
        edge_type = torch.index_select(orig_edge_type, 0, select_index_list)
        data.edge_index = edge_index
        data.edge_type = edge_type

    print(f'Epoch {i}, edge size {data.edge_index.size()} ')

    model = GAE(
            RGCNEncoder(data.num_nodes, 500, dataset.num_relations),
            DistMultDecoder(dataset.num_relations // 2, 500),
        ).to(device)

    model_list.append(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, train_epoch+1):
        loss = train(model, optimizer)
        print(f'Epoch: {epoch:05d}, Loss: {loss:.6f}')
        if (epoch % (train_epoch // 2)) == 0:
            valid_mrr, test_mrr = test(model)
            print(f'Val MRR: {valid_mrr:.4f}, Test MRR: {test_mrr:.6f}')
    
    test_train_res_list.append(test_mrr)

    if i == args.num_epochs:
        for old_model_i in range(args.num_epochs):
            valid_mrr, test_mrr = test(model_list[old_model_i])
            print(f'Test Acc in model {old_model_i}: {test_mrr:.6f}')

