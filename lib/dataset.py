import os
import sys
import random
import numpy as np
import json
import pandas as pd
import scipy
import datetime
import time

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csc_tensor, to_torch_csr_tensor
from sklearn.preprocessing import LabelEncoder

# https://snap.stanford.edu/data/com-Friendster.html
# https://snap.stanford.edu/data/twitter-2010.html
# https://snap.stanford.edu/data/sx-stackoverflow.html

def trans_nodes(root, dataset, output, split='\t'):
    node_remap = {}
    node_start = 0

    out_file = open(os.path.join(root, output), 'w')

    with open(os.path.join(root, dataset), 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            nodes = line.split(split)
            node_in = nodes[0]
            node_out = nodes[1]
            if node_in not in node_remap:
                node_remap[node_in] = node_start
                node_start += 1
            if node_out not in node_remap:
                node_remap[node_out] = node_start
                node_start += 1
            if len(nodes) == 2:
                out_file.write(str(node_remap[node_in]) + ' ' + str(node_remap[node_out]) + '\n')
            else:
                out_file.write(str(node_remap[node_in]) + ' ' + str(node_remap[node_out]) + ' ' + ' '.join(nodes[2:]) + '\n')

    print("Done", node_start)
    out_file.close()
    return node_start

def gen_reddit_dataset(root, node_in, feat_in, node_out, feat_out):
    node_remap = {}
    node_start = 0

    node_in = os.path.join(root, node_in)
    feat_in = os.path.join(root, feat_in)
    node_out = os.path.join(root, node_out)
    feat_out = os.path.join(root, feat_out)
    node_out_file = open(node_out, 'w')

    field = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
    node_df = pd.read_csv(node_in, delimiter='\t')[field]
    node_df[field[2]] = node_df[field[2]].apply(lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())))
    node_df = node_df.sort_values(by=field[2])
    for _, row in node_df.iterrows():
        if row[0] not in node_remap:
            node_remap[row[0]] = node_start
            node_start += 1
        if row[1] not in node_remap:
            node_remap[row[1]] = node_start
            node_start += 1
        node_out_file.write(str(node_remap[row[0]]) + ' ' + 
                            str(node_remap[row[1]]) + ' ' + 
                            str(row[2]) + '\n')
    x_df = pd.read_csv(feat_in, index_col=0, header=None)
    out_df = np.ones((node_start, x_df.shape[1]), dtype=np.float32) * np.mean(x_df.values, axis=0)
    for index, row in x_df.iterrows():
        if index in node_remap:
            out_df[node_remap[index],:] = row.values
    np.savetxt(feat_out, out_df, delimiter=',')
    return node_start


mmap_dataset = ['ogbn-papers100M', 'friendster', 'twitter']

tem_dataset = {'stackoverflow': ('stackoverflow-node.txt', 
                                 'sx-stackoverflow.txt'), 
               'reddit-hyperlink': ('soc-redditHyperlinks-node.txt', 
                                    'soc-redditHyperlinks.txt')}
reddit_dataset = {'reddit-body': ('soc-redditBody-node.txt',
                                  'soc-redditBody-x.csv', 
                                  'soc-redditHyperlinks-body.tsv'),
                  'reddit-title': ('soc-redditTitle-node.txt',
                                   'soc-redditTitle-x.csv',
                                   'soc-redditHyperlinks-title.tsv')}


class GraphDataset():

    def __init__(self, dataset: str, path='./data/dataset', batch_size=1000, num_epochs=10):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.dataset = dataset
        self.epochs = num_epochs
        self.root = path
        self.path = os.path.join(path, dataset.replace('-', '_'))
        if self.dataset in reddit_dataset:
            self.path = os.path.join(path, 'reddit_hyperlink')
        self.batch_size = batch_size
        self.orig_coo_file = os.path.join(self.path, 'orig_coo.dat') 
        self.begin_csr_file = (os.path.join(self.path, 'begin_csr_' + str(batch_size) +'.ind'), 
                               os.path.join(self.path, 'begin_csr_' + str(batch_size) +'.dat'))
        self.begin_csc_file = (os.path.join(self.path, 'begin_csc_' + str(batch_size) +'.ind'), 
                               os.path.join(self.path, 'begin_csc_' + str(batch_size) +'.dat'))
        # [batch][insert][delete] 
        self.update_lists_file = os.path.join(self.path, 'update_lists_' + str(batch_size) +'.dat')
        self.config_file = os.path.join(self.path, 'update_config_' + str(batch_size) +'.json')
        self.x_file = os.path.join(self.path, 'x.dat')


    def process(self):
        if self.dataset in mmap_dataset:
            config = {}
            if self.dataset == mmap_dataset[0]:
                dataset = PygNodePropPredDataset(name=self.dataset, root=self.root)
                data = dataset[0]
                config['num_features'] = int(dataset.num_features)
                config['num_classes'] = int(dataset.num_classes)
                config['num_nodes'] = int(data.num_nodes)
                data_edge_index = data.edge_index
                if os.path.isfile(self.x_file):
                    x = data.x.numpy().astype(np.float32)
                    x_mmap = np.memmap(self.x_file, mode='w+', 
                                        shape=x.shape, dtype=x.dtype)
                    np.copyto(x_mmap, x, casting='no')
                    x_mmap.flush()
            elif self.dataset == mmap_dataset[1]:
                config['num_nodes'] = 65608366
                config['num_features'] = 128
                config['num_classes'] = 50
                if not os.path.isfile(self.orig_coo_file):
                    nodes_file = 'com-friendster.ungraph-node.txt'
                    orig_file = os.path.join(self.path, nodes_file)
                    if not os.path.isfile(orig_file):
                        dataset_file = 'com-friendster.ungraph.txt'
                        config['num_nodes'] = trans_nodes(self.path, dataset_file, nodes_file)
                    else:
                        dataset_nodes = np.loadtxt(orig_file, dtype=np.int64)
                        np_nodes = np.array([dataset_nodes[:,0], dataset_nodes[:, 1]])
                        data_edge_index = torch.from_numpy(np_nodes)
            elif self.dataset == mmap_dataset[2]:
                config['num_features'] = 128
                config['num_classes'] = 50
                config['num_nodes'] = 41652230
                nodes = np.loadtxt(os.path.join(self.path, 'twitter-2010.txt'), 
                                   dtype=np.int64)
                np_nodes = np.array([nodes[:,0], nodes[:, 1]])
                data_edge_index = torch.from_numpy(np_nodes)
            else:
                raise ValueError('Unknown dataset')
            if os.path.isfile(self.orig_coo_file):
                orig_mmap = np.memmap(self.orig_coo_file, mode='r', dtype=np.int64).reshape(2, -1)
            else:
                print('Processing coo ...', file=sys.stderr)
                orig_edge_index = to_torch_coo_tensor(data_edge_index).indices().type(torch.int64)
                orig_mmap = np.memmap(self.orig_coo_file, mode='w+', shape=orig_edge_index.size(), 
                                  dtype=np.int64)
                np.copyto(orig_mmap, orig_edge_index, casting='no')
                orig_mmap.flush()
                print('Done!', file=sys.stderr)
            config['coo_size'] = tuple(orig_mmap.shape)
            config['coo_type'] = str(orig_mmap.dtype)
            print('Processing update ...', file=sys.stderr)
            update_mmap = np.memmap(self.update_lists_file, mode='w+', 
                                    shape=(self.epochs,  2, 2, self.batch_size // 2),
                                    dtype=np.int64)
            select_index = np.arange(orig_mmap.shape[-1])
            select_masks = np.ones_like(select_index, dtype=bool)
            insert_masks = np.ones_like(select_index, dtype=bool)
            for i in range(self.epochs):
                index = np.random.choice(select_index[select_masks], self.batch_size)
                select_masks[index] = False
                pos = len(index) // 2
                insert_list = index[:pos]
                insert_list.sort()
                np.copyto(update_mmap[self.epochs - 1 - i][0], 
                          orig_mmap[:, insert_list], casting='no')
                delete_list = index[pos:]
                delete_list.sort()
                np.copyto(update_mmap[self.epochs - 1 - i][1], 
                          orig_mmap[:, delete_list], casting='no')
                insert_masks[insert_list] = False
            update_mmap.flush()
            config['update_size'] = tuple(update_mmap.shape)
            config['update_type'] = str(update_mmap.dtype)
            print('Done!', file=sys.stderr)
            print('Processing csc ...', file=sys.stderr)
            coo = orig_mmap[:, insert_masks]
            num_nodes = config['num_nodes']
            v = np.ones_like(coo[0])
            coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
            graph_csc = coo.tocsc()
            indptr = graph_csc.indptr.astype(np.int64)
            indices = graph_csc.indices.astype(np.int64)
            csc_ind = np.memmap(self.begin_csc_file[0], mode='w+', 
                                shape=indptr.shape, dtype=indptr.dtype)
            csc_dat = np.memmap(self.begin_csc_file[1], mode='w+',
                                shape=indices.shape, dtype=indices.dtype)
            np.copyto(csc_ind, indptr, casting='no')
            np.copyto(csc_dat, indices, casting='no')
            csc_ind.flush()
            csc_dat.flush()
            config['csc_ind_size'] = tuple(csc_ind.shape)
            config['csc_ind_type'] = str(csc_ind.dtype)
            config['csc_dat_size'] = tuple(csc_dat.shape)
            config['csc_dat_type'] = str(csc_dat.dtype)
            print('Done!', file=sys.stderr)
            print('Processing csr...', file=sys.stderr)
            graph_csr = coo.tocsr()
            indptr = graph_csr.indptr.astype(np.int64)
            indices = graph_csr.indices.astype(np.int64)
            csr_ind = np.memmap(self.begin_csr_file[0], mode='w+', 
                                shape=indptr.shape, dtype=indptr.dtype)
            csr_dat = np.memmap(self.begin_csr_file[1], mode='w+',
                                shape=indices.shape, dtype=indices.dtype)
            np.copyto(csr_ind, indptr, casting='no')
            np.copyto(csr_dat, indices, casting='no')
            csr_ind.flush()
            csr_dat.flush()
            config['csr_ind_size'] = tuple(csr_ind.shape)
            config['csr_ind_type'] = str(csr_ind.dtype)
            config['csr_dat_size'] = tuple(csr_dat.shape)
            config['csr_dat_type'] = str(csr_dat.dtype)
            print('Done!', file=sys.stderr)
            json.dump(config, open(self.config_file, 'w'))
        elif self.dataset in tem_dataset:
            orig_file = os.path.join(self.path, tem_dataset[self.dataset][0])
            if not os.path.isfile(orig_file):
                num_nodes = trans_nodes(self.path, tem_dataset[self.dataset][1], 
                                        tem_dataset[self.dataset][0], ' ')
        elif self.dataset in reddit_dataset:
            # if not os.path.isfile(os.path.join(self.path, reddit_dataset[self.dataset][0])):
            num_nodes = gen_reddit_dataset(self.path, 
                                           reddit_dataset[self.dataset][2], 
                                           'web-redditEmbeddings-subreddits.csv',
                                           reddit_dataset[self.dataset][0], 
                                           reddit_dataset[self.dataset][1])


    def get_data(self, has_delete=False):
        ret = {}
        ret['num_edges'] = 0
        if self.dataset in mmap_dataset:
            if not os.path.isfile(self.config_file):
                self.process()
            self.recycle = []
            conf = json.load(open(self.config_file, 'r'))
            ret['num_features'] = conf['num_features']
            ret['num_classes'] = conf['num_classes']
            ret['num_nodes'] = conf['num_nodes']
            if os.path.isfile(self.x_file):
                x = np.memmap(self.x_file, mode='r', shape=(conf['num_nodes'], conf['num_features']), 
                              dtype=np.float32)
                self.recycle.append(x)
                ret['x'] = torch.from_numpy(x)
            csc_ind = np.memmap(self.begin_csc_file[0], mode='r', shape=tuple(conf['csc_ind_size']), 
                                dtype=conf['csc_ind_type'])
            self.recycle.append(csc_ind)
            csc_dat = np.memmap(self.begin_csc_file[1], mode='r', shape=tuple(conf['csc_dat_size']), 
                                dtype=conf['csc_dat_type'])
            self.recycle.append(csc_dat)
            ret['graph_csc'] = (torch.from_numpy(csc_ind), torch.from_numpy(csc_dat))
            csr_ind = np.memmap(self.begin_csr_file[0], mode='r', shape=tuple(conf['csr_ind_size']), 
                                dtype=conf['csr_ind_type'])
            self.recycle.append(csr_ind)
            csr_dat = np.memmap(self.begin_csr_file[1], mode='r', shape=tuple(conf['csr_dat_size']), 
                                dtype=conf['csr_dat_type'])
            self.recycle.append(csr_dat)
            ret['graph_csr'] = (torch.from_numpy(csr_ind), torch.from_numpy(csr_dat))
            update = np.memmap(self.update_lists_file, mode='r', shape=tuple(conf['update_size']), 
                               dtype=conf['update_type'])
            ret['update_lists'] = torch.from_numpy(update)
            if self.dataset is mmap_dataset[1]:
                ret['data_in_gpu'] = True
            else:
                ret['data_in_gpu'] = False
        elif self.dataset in tem_dataset or self.dataset in reddit_dataset:
            if self.dataset in tem_dataset:
                orig_file = os.path.join(self.path, tem_dataset[self.dataset][0])
                if self.dataset == 'stackoverflow':
                    ret['num_nodes'] = 2601977
                else:
                    ret['num_nodes'] = 67180
                ret['num_features'] = 128
                ret['num_classes'] = 64
            elif self.dataset in reddit_dataset:
                orig_file = os.path.join(self.path, reddit_dataset[self.dataset][0])
                np_x = np.loadtxt(os.path.join(self.path, reddit_dataset[self.dataset][1]), delimiter=',')
                ret['num_nodes'] = np_x.shape[0]
                ret['num_features'] = np_x.shape[1]
                ret['num_classes'] = 64
                ret['x'] = torch.from_numpy(np_x).type(torch.float32)
            nodes = np.loadtxt(orig_file, dtype=np.int64)
            np_nodes = np.array([nodes[:,0], nodes[:, 1]])
            data_edge_index = torch.from_numpy(np_nodes)
            # print('Processing coo...', file=sys.stderr)
            # orig_edge_index = to_torch_coo_tensor(data_edge_index).indices().type(torch.int64)
            # print(orig_edge_index.size())
            # orig_csr_index = to_torch_csr_tensor(orig_edge_index).crow_indices().type(torch.int64)
            print('Processing update...', file=sys.stderr)
            timestamps = nodes[:, 2]
            start_time = datetime.datetime.fromtimestamp(timestamps[0]).replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = datetime.datetime.fromtimestamp(timestamps[-1]).replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
            start_unixtime = time.mktime(start_time.timetuple())
            deta_unixtime = time.mktime((start_time + datetime.timedelta(days=1)).timetuple()) - start_unixtime
            end_unixtime = time.mktime(end_time.timetuple())
            delete_counts = torch.zeros(self.epochs, dtype=torch.int64)
            if has_delete:
                for t in timestamps:
                    ix = int((t - start_unixtime) / deta_unixtime)
                    if ix < self.epochs:
                        delete_counts[ix] += 1
                    else:
                        break
            delete_index = torch.zeros(self.epochs + 1, dtype=torch.int64)
            torch.cumsum(delete_counts, 0, out=delete_index[1:])
            insert_counts = torch.zeros(self.epochs, dtype=torch.int64)
            for t in timestamps[::-1]:
                ix = int((end_unixtime - t) / deta_unixtime)
                if ix < self.epochs:
                    insert_counts[ix] += 1
                else:
                    break
            insert_index = torch.zeros(self.epochs + 1, dtype=torch.int64)
            torch.cumsum(insert_counts, 0, out=insert_index[1:])
            insert_index = (len(timestamps) - insert_index).flip(0)
            update_torch = []
            # remove_masks = torch.ones_like(orig_edge_index[0], dtype=torch.bool)
            for i in range(self.epochs):
                insert_coo = data_edge_index[:, insert_index[i]:insert_index[i+1]]
                if insert_coo.size()[-1] == 0:
                    insert_coo = torch.zeros((2, 0), dtype=torch.int64)
                else:
                    insert_coo = to_torch_coo_tensor(insert_coo).indices().type(torch.int64)
                delete_coo = data_edge_index[:, delete_index[i]:delete_index[i+1]]
                if delete_coo.size()[-1] == 0:
                    delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                else:
                    delete_coo = to_torch_coo_tensor(delete_coo).indices().type(torch.int64)
                update_torch.append((insert_coo, delete_coo))
                print(f'update {i}: insert {insert_coo.size(1)} delete {delete_coo.size(1)}')
                # for v, u in zip(insert_coo[0], insert_coo[1]):
                #     for j_index in range(orig_csr_index[v], orig_csr_index[v+1]):
                #         if orig_edge_index[1][j_index] == u:
                #             remove_masks[j_index] = False
                #             break
            ret['update_lists'] = update_torch
            print('Processing csr csc ...', file=sys.stderr)
            # coo = orig_edge_index[:, remove_masks]
            coo = data_edge_index[:, :insert_index[0]]
            self.gen_coo(ret, coo)
        elif self.dataset == 'GDELT':
            print('Processing x y ...', file=sys.stderr)
            x = torch.load(os.path.join(self.path, 'node_features.pt')).type(torch.float32)
            ret['x'] = x
            ret['num_nodes'] = x.size(0)
            ret['num_features'] = x.size(1)
            y = np.loadtxt(os.path.join(self.path, 'labels.txt'), dtype=np.int64)
            ret['num_classes'] = np.max(y[:, 1]) + 1
            np_y = torch.from_numpy(np.array([y[:,0], y[:, 1], y[:, 2]]))
            print('Processing coo...', file=sys.stderr)
            nodes = np.loadtxt(os.path.join(self.path, 'edges.txt'), dtype=np.int64)
            np_nodes = np.array([nodes[:,0], nodes[:, 1]])
            data_edge_index = torch.from_numpy(np_nodes)
            print('Processing update...', file=sys.stderr)
            timestamps = nodes[:, 2]
            start_unixtime = np.min(timestamps)
            end_unixtime = np.max(timestamps) + 1
            period = 1
            update_torch = []
            update_label = []
            for i in range(self.epochs):
                t_start = end_unixtime - (self.epochs - i) * period
                t_end = t_start + period
                time_mask = np.logical_and(timestamps >= t_start, timestamps < t_end)
                insert_coo = data_edge_index[:, time_mask]
                if insert_coo.size()[-1] == 0:
                    insert_coo = torch.zeros((2, 0), dtype=torch.int64)
                else:
                    insert_coo = to_torch_coo_tensor(insert_coo).indices().type(torch.int64)
                label_mask = np.logical_and(np_y[2] >= t_start, np_y[2] < t_end)
                update_label.append(np_y[:, label_mask])
                if has_delete:
                    d_start = start_unixtime + i * period
                    d_end = d_start + period
                    time_mask = np.logical_and(timestamps >= d_start, timestamps < d_end)
                    delete_coo = data_edge_index[:, time_mask]
                    if delete_coo.size()[-1] == 0:
                        delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                    else:
                        delete_coo = to_torch_coo_tensor(delete_coo).indices().type(torch.int64)
                else:
                    delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                update_torch.append((insert_coo, delete_coo))
                print(f'update {i}: insert {insert_coo.size(1)} t_start {t_start} t_end {t_end} end_unixtime {end_unixtime}\n\tdelete {delete_coo.size(1)}', f'd_start {d_start} d_end {d_end} start_unixtime {start_unixtime}' if has_delete else '')
            ret['update_lists'] = update_torch
            ret['update_labels'] = update_label
            # np_y = np_y[:, np_y[2] < (end_unixtime - self.epochs * period)]
            # df_y = pd.DataFrame(np_y.T, columns = ['node', 'label', 'time'])
            # df_y_index = df_y.groupby('node')['time'].max().index
            ret['train_labels'] = np_y[:2, np_y[2] < (end_unixtime - self.epochs * period)]
            print('Processing csr csc ...', file=sys.stderr)
            coo = data_edge_index[:, timestamps < (end_unixtime - self.epochs * period)]
            self.gen_coo(ret, coo)
        elif self.dataset == 'ogbn-arxiv':
            time_file = os.path.join(self.path, 'raw', 'node_year.csv.gz')
            years = torch.from_numpy(pd.read_csv(time_file, compression='gzip', header = None).values).to(torch.int64).squeeze()
            dataset = PygNodePropPredDataset(name=self.dataset, root=self.root)
            data = dataset[0]
            ret['x'] = data.x.to(torch.float32)
            ret['num_features'] = dataset.num_features
            ret['num_classes'] = dataset.num_classes
            ret['num_nodes'] = data.num_nodes
            ret['num_edges'] = data.num_edges
            label_y = data.y.to(torch.int64)
            orig_edge_index = to_torch_coo_tensor(data.edge_index).indices().type(torch.int64)
            update_file = os.path.join(self.path, f'update_lists_{self.epochs}.dat')
            if os.path.isfile(update_file):
                update_torch, update_label, insert_masks, train_label = torch.load(update_file)
                for i_update, (insert_coo, delete_coo) in enumerate(update_torch):
                    print(f'update {i_update}: insert {insert_coo.size(1)} delete {delete_coo.size(1)}')
            else:
                update_torch = []
                update_label = []
                insert_masks = torch.ones(orig_edge_index.size(1), dtype=torch.bool)
                for i in range(self.epochs):
                    t_start = years.max() - (self.epochs - 1 - i)
                    curr_year = (years == t_start).nonzero()[:,0].to(dtype=torch.int64).squeeze()
                    curr_masks = torch.zeros_like(insert_masks, dtype=torch.bool)
                    if curr_year.nelement() == 0:
                        pass
                        # raise ValueError('No data in year')
                    elif curr_year.nelement() == 1:
                        curr_masks[orig_edge_index[0] == curr_year] = True
                    else:
                        for year in curr_year:
                            curr_masks[orig_edge_index[0] == year] = True
                    insert_coo = orig_edge_index[:, curr_masks]
                    insert_masks[curr_masks] = False
                    if has_delete:
                        d_start = years.min() + i
                        d_year = (years == d_start).nonzero()[:,0].to(dtype=torch.int64).squeeze()
                        d_masks = torch.zeros_like(insert_masks, dtype=torch.bool)
                        if d_year.nelement() == 0:
                            pass
                            # raise ValueError('No data in year')
                        elif d_year.nelement() == 1:
                            d_masks[orig_edge_index[0] == d_year] = True
                        else:
                            for year in d_year:
                                d_masks[orig_edge_index[0] == year] = True
                        delete_coo = orig_edge_index[:, d_masks]
                    else:
                        delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                    update_torch.append((insert_coo, delete_coo))
                    update_label.append(torch.stack([curr_year, label_y[curr_year].view(-1)]))
                    print(f'update {i}: insert {insert_coo.size(1)} t_start {t_start}\n\tdelete {delete_coo.size(1)}', f'd_start {d_start}' if has_delete else '')
                train_year = (years < years.max() - (self.epochs - 1)).nonzero()[:,0].to(dtype=torch.int64).squeeze()
                train_label = torch.stack([train_year, label_y[train_year].view(-1)])
                torch.save((update_torch, update_label, insert_masks, train_label), update_file)
            ret['update_lists'] = update_torch
            ret['update_labels'] = update_label
            ret['train_labels'] = train_label
            print('Processing csr csc ...', file=sys.stderr)
            coo = orig_edge_index[:, insert_masks]
            self.gen_coo(ret, coo)
        elif self.dataset == 'ogbl-collab':
            time_file = os.path.join(self.path, 'raw', 'edge_year.csv.gz')
            years = torch.from_numpy(pd.read_csv(time_file, compression='gzip', header = None).values).to(torch.int64).squeeze()
            dataset = PygLinkPropPredDataset(name=self.dataset, root=self.root)
            data = dataset[0]
            ret['x'] = data.x.to(torch.float32)
            ret['num_features'] = dataset.num_features
            ret['num_classes'] = 64
            ret['num_nodes'] = data.num_nodes
            orig_edge_index = data.edge_index[:, :years.nelement()].type(torch.int64)
            update_torch = []
            for i in range(self.epochs):
                t_start = years.max() - (self.epochs - 1 - i)
                insert_coo = orig_edge_index[:, years == t_start]
                if insert_coo.nelement() == 0:
                    raise ValueError('No data in year')
                if has_delete:
                    d_time = years.min() + i
                    delete_coo = orig_edge_index[:, years == d_time]
                    if delete_coo.nelement() == 0:
                        raise ValueError('No data in year')
                else:
                    delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                update_torch.append((insert_coo, delete_coo))
                print(f'update {i}: insert {insert_coo.size(1)} t_start {t_start}\n\tdelete {delete_coo.size(1)}', f'd_start {d_time}' if has_delete else '')
            ret['update_lists'] = update_torch
            print('Processing csr csc ...', file=sys.stderr)
            coo = orig_edge_index[:, years < years.max() - (self.epochs - 1)]
            coo = to_torch_coo_tensor(coo).indices().type(torch.int64)
            self.gen_coo(ret, coo)
        elif self.dataset == 'elliptic':
            nodes = pd.read_csv(os.path.join(self.path, 'elliptic_txs_features.csv'), header=None)
            edges = pd.read_csv(os.path.join(self.path, 'elliptic_txs_edgelist.csv'))
            classes = pd.read_csv(os.path.join(self.path, 'elliptic_txs_classes.csv'))

            # Create a mapping from transaction IDs to node indices
            tx_id_to_index = {tx_id: idx for idx, tx_id in enumerate(nodes[0])}
            valid_edges = edges[edges['txId1'].isin(tx_id_to_index) & edges['txId2'].isin(tx_id_to_index)]
            valid_edges['txId1'] = valid_edges['txId1'].map(tx_id_to_index)
            valid_edges['txId2'] = valid_edges['txId2'].map(tx_id_to_index)
            orig_edge_index = torch.tensor(valid_edges.values.T, dtype=torch.int64)

            le = LabelEncoder()
            label_y = torch.tensor(le.fit_transform(classes['class']), dtype=torch.int64)

            timestamps = torch.tensor(nodes.iloc[:, 1].values, dtype=torch.int64)
            features = torch.tensor(nodes.iloc[:, 2:].values, dtype=torch.float32)
            ret['x'] = features
            ret['num_nodes'] = features.size(0) # 203769
            ret['num_features'] = features.size(1) # 165
            ret['num_classes'] = 3
            ret['num_edges'] = orig_edge_index.size(1) # 234355
    
            update_file = os.path.join(self.path, f'update_lists_{self.epochs}.dat')
            if os.path.isfile(update_file):
                update_torch, update_label, insert_masks, train_label = torch.load(update_file)
                for i_update, (insert_coo, delete_coo) in enumerate(update_torch):
                    print(f'update {i_update}: insert {insert_coo.size(1)} delete {delete_coo.size(1)}')
            else:
                update_torch = []
                update_label = []
                insert_masks = torch.ones(orig_edge_index.size(1), dtype=torch.bool)
                for i in range(self.epochs):
                    t_start = timestamps.max() - (self.epochs - 1 - i)
                    curr_step = (timestamps == t_start).nonzero()[:, 0].to(dtype=torch.int64).squeeze()
                    curr_masks = torch.zeros_like(insert_masks, dtype=torch.bool)
                    if curr_step.nelement() == 0:
                        pass
                    elif curr_step.nelement() == 1:
                        curr_masks[orig_edge_index[0] == curr_step] = True
                    else:
                        for step in curr_step:
                            curr_masks[orig_edge_index[0] == step] = True
                    insert_coo = orig_edge_index[:, curr_masks]
                    insert_masks[curr_masks] = False
                    if has_delete:
                        d_start = timestamps.min() + i
                        d_step = (timestamps == d_start).nonzero()[:, 0].to(dtype=torch.int64).squeeze()
                        d_masks = torch.zeros_like(insert_masks, dtype=torch.bool)
                        if d_step.nelement() == 0:
                            pass
                        elif d_step.nelement() == 1:
                            d_masks[orig_edge_index[0] == d_step] = True
                        else:
                            for step in d_step:
                                d_masks[orig_edge_index[0] == step] = True
                        delete_coo = orig_edge_index[:, d_masks]
                    else:
                        delete_coo = torch.zeros((2, 0), dtype=torch.int64)
                    update_torch.append((insert_coo, delete_coo))
                    update_label.append(torch.stack([curr_step, label_y[curr_step].view(-1)]))
                    print(f'update {i}: insert {insert_coo.size(1)} t_start {t_start}\n\tdelete {delete_coo.size(1)}', f'd_start {d_start}' if has_delete else '')
                train_step = (timestamps < timestamps.max() - (self.epochs - 1)).nonzero()[:, 0].to(dtype=torch.int64).squeeze()
                train_label = torch.stack([train_step, label_y[train_step].view(-1)])
                torch.save((update_torch, update_label, insert_masks, train_label), update_file)
            ret['update_lists'] = update_torch
            ret['update_labels'] = update_label
            ret['train_labels'] = train_label
            print('Processing csr csc...', file=sys.stderr)
            coo = orig_edge_index[:, insert_masks]
            self.gen_coo(ret, coo)
        else:
            if self.dataset == 'ogbn-products':
                dataset = PygNodePropPredDataset(name=self.dataset, root=self.root)
            elif self.dataset == 'reddit':
                dataset = Reddit(self.root)
            else:
                raise ValueError('Unknown dataset')
            data = dataset[0]
            ret['x'] = data.x.to(torch.float32)
            ret['num_features'] = dataset.num_features
            ret['num_classes'] = dataset.num_classes
            ret['num_nodes'] = data.num_nodes
            print('Processing coo...', file=sys.stderr)
            orig_edge_index = to_torch_coo_tensor(data.edge_index).indices().type(torch.int64)
            print('Processing update...', file=sys.stderr)
            update_torch = torch.zeros((self.epochs,  2, 2, self.batch_size // 2),
                                    dtype=torch.int64)
            select_index = torch.arange(orig_edge_index.size()[-1])
            select_masks = torch.ones_like(select_index, dtype=torch.bool)
            insert_masks = torch.ones_like(select_index, dtype=torch.bool)
            for i in range(self.epochs):
                index = np.random.choice(select_index[select_masks], self.batch_size)
                select_masks[index] = False
                pos = len(index) // 2
                insert_list = index[:pos]
                insert_list.sort()
                update_torch[self.epochs - 1 - i][0] = orig_edge_index[:, insert_list]
                delete_list = index[pos:]
                delete_list.sort()
                update_torch[self.epochs - 1 - i][1] = orig_edge_index[:, delete_list]
                insert_masks[insert_list] = False
            ret['update_lists'] = update_torch
            print('Processing csr csc ...', file=sys.stderr)
            coo = orig_edge_index[:, insert_masks]
            self.gen_coo(ret, coo)

        return ret

    def gen_coo(self, ret, coo):
        ret['graph_coo'] = coo
        csc = to_torch_csc_tensor(coo)
        csc_ind = csc.ccol_indices()
        remain = ret['num_nodes'] + 1 - csc_ind.size(0)
        if remain > 0:
            csc_app = torch.ones(remain, dtype=torch.int64) * csc_ind[-1]
            csc_ind = torch.cat([csc_ind, csc_app], 0)
        ret['graph_csc'] = (csc_ind, csc.row_indices())
        csr = to_torch_csr_tensor(coo)
        csr_ind = csr.crow_indices()
        remain = ret['num_nodes'] + 1 - csr_ind.size(0)
        if remain > 0:
            csr_app = torch.ones(remain, dtype=torch.int64) * csr_ind[-1]
            csr_ind = torch.cat([csr_ind, csr_app], 0)
        ret['graph_csr'] = (csr_ind, csr.col_indices())
        ret['data_in_gpu'] = True


    def close(self):
        if self.dataset in mmap_dataset:
            if self.recycle is not None:
                for i in self.recycle:
                    i._mmap.close()
        return None


if __name__ == '__main__':
    name = 'GDELT'
    if len(sys.argv) > 1:
        name = sys.argv[1]
    dataset = GraphDataset(name, path='/home/jiangqsh/jqs/GNNDrive/data/dataset', batch_size=1000)
    # dataset = GraphDataset(name, path='/home/cuihj/gnn-inference/data/dataset', batch_size=1000)
    # dataset = GraphDataset('ogbn-papers100M', path='../data/dataset')
    # dataset.process()
    data = dataset.get_data()
    print(data)
