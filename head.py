""" GNN heads are the last layer of a GNN right before loss computation.

They are constructed in the init function of the gnn.GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


########### Head ############

class GNNNodeHead(nn.Module):
    '''Head of GNN, node prediction'''

    def __init__(self, dim_in):
        super(GNNNodeHead, self).__init__()

    def forward(self, emb, device, node_index=None, train=False):
        if train:
           pred = emb.to(device)
        else:
            pred = emb[node_index].to(device)
            pred = pred.argmax(dim=-1)
        return pred


class GNNEdgeHead(nn.Module):
    r"""The GNN head module for edge prediction tasks. This module takes a (batch of) graphs and
    outputs ...
    """

    def __init__(self, dim_in: int):
        ''' Head of Edge and link prediction models.

        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        '''
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(GNNEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        self.W1 = nn.Linear(dim_in * 2, dim_in)
        self.W2 = nn.Linear(dim_in, 1)

    def forward(self, emb, device, edge_index):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        h = torch.cat([src, dst], dim=1).to(device)
        score = self.W2(F.relu(self.W1(h))).squeeze(1)
        return score


class GNNEdgeHead2(nn.Module):

    def __init__(self, dim_in: int):
        super(GNNEdgeHead2, self).__init__()
        # module to decode edges from node embeddings
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, emb, device, edge_index):
        num_edge = edge_index.size(1) // 2
        src = emb[edge_index[0]].to(device)
        dst = emb[edge_index[1]].to(device)

        h_pos_src = self.src_fc(src[:num_edge])
        h_neg_src = self.dst_fc(src[num_edge:])
        h_pos_dst = self.src_fc(dst[:num_edge]) 
        h_neg_dst = self.dst_fc(dst[num_edge:])

        h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_neg_src + h_neg_dst)
        score = torch.cat([self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)]).squeeze(1)
        return score


# Head models for external interface
head_dict = {
    'node': GNNNodeHead,
    'edge': GNNEdgeHead,
    'link_pred': GNNEdgeHead,
}
