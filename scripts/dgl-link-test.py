"""
Link Prediction using Graph Neural Networks
===========================================

In the :doc:`introduction <1_introduction>`, you have already learned
the basic workflow of using GNNs for node classification,
i.e. predicting the category of a node in a graph. This tutorial will
teach you how to train a GNN for link prediction, i.e. predicting the
existence of an edge between two arbitrary nodes in a graph.

By the end of this tutorial you will be able to

-  Build a GNN-based link prediction model.
-  Train and evaluate the model on a small DGL-provided dataset.

(Time estimate: 28 minutes)

"""

import itertools
import os
import random
import sys

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

data_opt = int(sys.argv[1])

if data_opt == 1:
    dataset = dgl.data.CiteseerGraphDataset()
elif data_opt == 2:
    dataset = dgl.data.PubmedGraphDataset()
else:
    dataset = dgl.data.CoraGraphDataset()

g = dataset[0]
random.seed(0)
np.random.seed(0)

######################################################################
# Prepare training and testing sets
# ---------------------------------
#
# This tutorial randomly picks 10% of the edges for positive examples in
# the test set, and leave the rest for the training set. It then samples
# the same number of edges for negative examples in both sets.
#

# Split edge set for training and testing
u, v = g.edges()

h_u = u[:int(len(u) / 2)]
h_v = v[:int(len(v) / 2)]
h_g = dgl.graph((h_u, h_v), num_nodes=g.num_nodes())

eids = np.arange(h_g.num_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.2)

test_pos_u, test_pos_v = h_u[eids[:test_size]], h_v[eids[:test_size]]
train_pos_u, train_pos_v = h_u[eids[test_size:]], h_v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
neg_u, neg_v = np.where(adj_neg != 0)
neg_num = min(len(neg_u), len(h_u))
neg_eids = np.random.choice(len(neg_u), neg_num)
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
    )
train_neg_u, train_neg_v = (
        neg_u[neg_eids[test_size:]],
        neg_v[neg_eids[test_size:]],
    )

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

rest_g = dgl.remove_edges(g, eids[:test_size])
defind_index = np.arange(rest_g.num_edges())
defind_index = np.random.permutation(defind_index)

def get_train(percent=1.0):
    if percent < 1.0:
        pos = int(len(defind_index) * percent)
        train_g = dgl.remove_edges(rest_g, defind_index[pos:])
    else:
        train_g = rest_g

    return train_g


from dgl.nn import SAGEConv


# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


from sklearn.metrics import roc_auc_score

models = []

for percent in range(50, 101, 5):
    percent /= 100
    train_g = get_train(percent)
    print("Train size:", train_g.num_edges(), train_pos_g.num_edges(), train_neg_g.num_edges(), percent)

    # ----------- 3. set up model, loss and optimizer -------------- #
    model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
    pred = DotPredictor()

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), pred.parameters()), lr=0.01
        )

    all_logits = []
    for e in range(100):
        # forward
        h = model(train_g, train_g.ndata["feat"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if e % 5 == 0:
        #     print("In epoch {}, loss: {}".format(e, loss))

    # ----------- 5. check results ------------------------ #
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print("AUC in round", len(models), ',', compute_auc(pos_score, neg_score))
    # print("")
    models.append((model, pred, h))

    if percent >= 1.0:
        for im in range(len(models) - 1):
            with torch.no_grad():
                old_model = models[im][0]
                old_pred = models[im][1]
                old_h = models[im][2]
                pos_score = old_pred(test_pos_g, old_h)
                neg_score = old_pred(test_neg_g, old_h)
                print("AUC in", im, ',', compute_auc(pos_score, neg_score))
                old_hh = old_model(train_g, train_g.ndata["feat"])
                pos_score = old_pred(test_pos_g, old_hh)
                neg_score = old_pred(test_neg_g, old_hh)
                print("AUC in", im, ',', compute_auc(pos_score, neg_score))
