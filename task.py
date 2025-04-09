import itertools
import os
import random
import torch

from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.utils import to_torch_coo_tensor

from head import head_dict
from model_layer import SAGE,GIN,GCN


def train_link(model, pred, optimizer, loader, num_nodes, device):
    model.train()
    pred.train()

    total_loss = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        out = pred(out, device, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    loss = total_loss / num_nodes
    return loss

def train_node(model, pred, optimizer, loader, train_labels, device):
    model.train()
    pred.train()

    total_loss = total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch_size]
        out = pred(out, device, train=True)
        end_index = min((i+1)*batch_size, len(train_labels[1]))
        label = train_labels[1][i*batch_size:end_index].to(device)
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == label).sum())
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

train_func = {'link_pred': train_link, 'node': train_node}

def negative_sampling(edge_index, num_nodes, insert_edges, ratio=1.0):
    num_edges = insert_edges.size(1)
    num_edges = round(ratio * num_edges) if num_edges > 0 else 1000

    edge_index = to_torch_coo_tensor(edge_index).indices()

    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ),
                                              device=neg_edge_index.device)
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ),
                                              device=neg_edge_index.device)
    
    num_edges = min(num_edges, neg_edge_index.size(1))
    select_index = torch.tensor(random.sample(
        range(neg_edge_index.size(1)), num_edges), 
        device=neg_edge_index.device)

    if insert_edges.size(1) == 0:
        insert_edges = edge_index[:, select_index]

    assert insert_edges.size(1) == num_edges
    eval_index = torch.cat([insert_edges, neg_edge_index[:, select_index]], dim=1)
    eval_label = torch.cat([torch.ones(insert_edges.size(1)), torch.zeros(num_edges)], dim=0)

    return eval_index, eval_label

train_epoch = {'link_pred': 100, 'node': 100}
prefix = {'link_pred': 'edge', 'node': 'node'}
path_tmp = './models/model_data_%s_%s.pt'


# Define model
def new_model(model_name, num_features, num_hiddens, num_classes, sizes):
    if model_name == 'sage':
        model = SAGE(num_features, num_hiddens, num_classes, num_layers=len(sizes))
    elif model_name == 'gcn':
        model = GCN(num_features, num_hiddens, num_classes, num_layers=len(sizes))
    elif model_name == 'gin':
        model = GIN(num_features, num_hiddens, num_classes, num_layers=len(sizes))
    else:
        raise NotImplementedError
    return model


class Task:

    def __init__(self, task, model, num_classes, batch_size, dataset, device, gpu_infer=True):
        random.seed(0)
        self.task = task
        self.device = device
        self.model = model
        self.pred = head_dict[task](num_classes)
        self.batch_size = batch_size
        self.lr = 0.01
        self.train_epoch = train_epoch[task]
        self.num_workers = 10
        self.model_path = path_tmp % (prefix[task], dataset)
        self.gpu_infer = gpu_infer

    def init_models(self, data, sizes, new_model=False, train_labels=None):
        if new_model:
            if self.task == 'link_pred':
                assert train_labels is None
                train_labels = data.num_nodes
                optimizer = torch.optim.Adam(
                        itertools.chain(self.model.parameters(), self.pred.parameters()), 
                        lr=self.lr, weight_decay=0.0005)
                train_loader = LinkNeighborLoader(data,
                                            batch_size=self.batch_size // 2,
                                            # shuffle=True,
                                            neg_sampling_ratio=1.0,
                                            num_neighbors=sizes,
                                            num_workers=self.num_workers)
            else:
                assert train_labels is not None
                optimizer = torch.optim.Adam(
                        self.model.parameters(), 
                        lr=self.lr, weight_decay=0.0005)
                train_loader = NeighborLoader(data,
                                            batch_size=self.batch_size,
                                            # shuffle=True,
                                            input_nodes=train_labels[0],
                                            num_neighbors=sizes,
                                            num_workers=self.num_workers)

            self.model = self.model.to(self.device)
            self.pred = self.pred.to(self.device)
            for epoch in range(self.train_epoch):
                output = train_func[self.task](self.model, self.pred, optimizer, train_loader, train_labels, self.device)
                if (epoch+1) % 10 == 0:
                    if self.task == 'link_pred':
                        loss = output
                        print(f'    Epoch: {epoch:02d}, Loss: {loss:.6f}')
                    else:
                        loss, acc = output
                        print(f'    Epoch: {epoch:02d}, Loss: {loss:.6f}, Acc: {acc:.6f}')

            torch.save([self.model.state_dict(), self.pred.state_dict()], self.model_path) # NOTE
        else:
            model_data = torch.load(self.model_path, map_location='cpu')
            load_ret = self.model.load_state_dict(model_data[0])
            assert len(load_ret[0]) == 0 and len(load_ret[1]) == 0
            self.model = self.model.to(self.device)
            self.model.eval()

            load_ret = self.pred.load_state_dict(model_data[1])
            assert len(load_ret[0]) == 0 and len(load_ret[1]) == 0
            self.pred = self.pred.to(self.device)
            self.pred.eval()

    @torch.no_grad()
    def inference(self, data):
        if self.gpu_infer:
            infer_device = self.device
        else:
            infer_device = 'cpu'
            self.model = self.model.to(infer_device)
        self.model.eval()
        loader = NeighborLoader(data, num_neighbors=[-1],
                                input_nodes=torch.arange(data.num_nodes),
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)
        feat = self.model.inference(loader, device=infer_device)
        self.model = self.model.to(self.device)
        return feat #.to(self.device)

    @torch.no_grad()
    def test(self, data, test_obj, node_emb=None):
        if self.task == 'link_pred':
            test_edges = test_obj[0]
            eval_index, eval_label = negative_sampling(torch.cat([data.edge_index, test_edges], dim=1), data.num_nodes, test_edges)
        else:
            eval_index = test_obj[0]
            eval_label = test_obj[1]

        self.model.eval()
        self.pred.eval()
        if node_emb is None:
            node_emb = self.inference(data)
        out = self.pred(node_emb, self.device, eval_index)

        if self.task == 'link_pred':
            pred_score = torch.sigmoid(out)
            auc = roc_auc_score(eval_label.cpu().numpy(), pred_score.cpu().numpy())
            print(f'    AUC: {auc}')
            return auc, eval_index, eval_label
        else:
            acc = int((out == eval_label.to(self.device)).sum()) / eval_label.nelement()
            f1 = f1_score(eval_label.cpu(), out.cpu(), average='micro')
            f1_w = f1_score(eval_label.cpu(), out.cpu(), average='weighted')
            print(f'    ACC: {acc}, F1 {f1}, F1_w {f1_w}')
            return acc, f1, f1_w
        

class Downstream:
    def __init__(self, task, model_name, dataset, layout, sizes, device, labels=None):
        self.task = task
        self.model_name = model_name
        self.model = new_model(model_name, layout[0], layout[1], layout[2], sizes)
        self.device = device
        self.dataset = dataset
        
        if model_name == 'sage' and task != 'none':
            model_path = path_tmp % (prefix[task], dataset)
            model_data = torch.load(model_path, map_location='cpu')
            load_ret = self.model.load_state_dict(model_data[0])
            assert len(load_ret[0]) == 0 and len(load_ret[1]) == 0
        self.model = self.model.to(self.device)
        self.model.eval()

        if task != 'none':
            self.pred = head_dict[task](layout[2])
            if model_name == 'sage':
                load_ret = self.pred.load_state_dict(model_data[1])
                assert len(load_ret[0]) == 0 and len(load_ret[1]) == 0
            self.pred = self.pred.to(self.device)
            self.pred.eval()
            if labels is None:
                self.eval_data = torch.load(f'/home/jiangqsh/nus/roland/eval_edge_data_{self.dataset}.pt')
                # self.eval_data = torch.load(f'/home/cuihj/roland/eval_edge_data_{self.dataset}.pt')
            else:
                self.eval_data = labels

    def init_embs(self, feature, edge_index, num_nodes, batch_size, on_gpu=True):
        if self.task == 'none':
            embs = []
            aggrs = []
            for layer_i in range(self.model.num_layers):
                aggrs.append(torch.rand((num_nodes, self.model.convs[layer_i].in_channels), dtype=torch.float32))
                embs.append(torch.rand((num_nodes, self.model.convs[layer_i].out_channels), dtype=torch.float32))
            return embs, aggrs

        emb_path = f'./models/emb_data_{self.task}_{self.dataset}.pt' #_{self.model_name}
        if os.path.isfile(emb_path):
            init_embs, init_aggrs = torch.load(emb_path)
            return init_embs, init_aggrs

        infer_dev = self.device
        if not on_gpu:
            infer_dev = 'cpu'
            self.model = self.model.to(infer_dev)
        init_embs, init_aggrs = self.model.inference(feature, edge_index, num_nodes, batch_size, 10, infer_dev)
        if not on_gpu:
            self.model = self.model.to(self.device)
        torch.save((init_embs, init_aggrs), emb_path)
        return init_embs, init_aggrs

    @torch.no_grad()
    def test(self, i_epoch, node_emb, is_edge=False):
        if self.task == 'none':
            return 0

        eval_index = self.eval_data[i_epoch][0]
        eval_label = self.eval_data[i_epoch][1]

        self.model.eval()
        self.pred.eval()
        out = self.pred(node_emb, self.device, eval_index)

        if is_edge:
            pred_score = torch.sigmoid(out)
            auc = roc_auc_score(eval_label.cpu().numpy(), pred_score.cpu().numpy())
            return auc
        else:
            # acc = int((out == eval_label.to(self.device)).sum()) / eval_label.nelement()
            f1 = f1_score(eval_label.cpu(), out.cpu(), average='micro')
            # f1_w = f1_score(eval_label.cpu(), out.cpu(), average='weighted')
            return f1 # acc , f1, f1_w
