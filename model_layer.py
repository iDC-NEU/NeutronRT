from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GCNConv, MLP, GINConv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import spmm, scatter
from torch_geometric.loader import NeighborSampler


class Basic(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(Basic, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.init_conv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_channels, hidden_channels))
        self.convs.append(self.init_conv(hidden_channels, out_channels))

    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x, _ = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x
    
    @torch.no_grad()
    def layer_wise(self, layer_i, x, edge_index, size):
        x_target = x[:size]
        y, aggr_y = self.convs[layer_i]((x, x_target), edge_index)
        if layer_i != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y, aggr_y
    
    @torch.no_grad()
    def layer_wise_partly(self, layer_i, new_x, x_target):
        y, _ = self.convs[layer_i]((new_x, x_target), None)
        if layer_i != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y
    
    def get_aggr(self, layer_i):
        return self.convs[layer_i].aggr
    
    @torch.no_grad()
    def inference(self, x, edge_index, num_nodes, batch_size, num_workers, device):
        loader = NeighborSampler(edge_index, 
                                 node_idx=torch.arange(num_nodes),
                                 sizes=[-1], 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers)
        embs = []
        aggrs = []
        for layer_i in range(self.num_layers):
            aggrs.append(torch.zeros((num_nodes, self.convs[layer_i].in_channels)))
            embs.append(torch.zeros((num_nodes, self.convs[layer_i].out_channels)))
            for step, (batch_size, ids, adjs) in enumerate(loader):
                if layer_i == 0:
                    update_x = x[ids].to(device)
                else:
                    update_x = embs[layer_i-1][ids].to(device)
                sampled_edge_index = adjs[0].to(device)
                update_y, update_a = self.layer_wise(layer_i, update_x, sampled_edge_index, batch_size)
                nodes = ids[:batch_size]
                embs[layer_i][nodes] = update_y.cpu()
                aggrs[layer_i][nodes] = update_a.cpu()

        return embs, aggrs

class SAGEConvReduce(SAGEConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, aggr, **kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, None],
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        x_r = x[1]
        if edge_index is None:
            aggr_out = self.update(x[0])
        else:
            # propagate_type: (x: OptPairTensor)
            aggr_out = self.propagate(edge_index, x=x)
            # alernaive way to do this
            # coo = torch.sparse_coo_tensor(torch.stack([edge_index[1], edge_index[0]]), 
            #     torch.ones_like(edge_index[0], dtype=torch.float32), 
            #     size=(x_r.size(0), out.size(0)),
            #     device=out.device)
            # out = spmm(coo, out, self.aggr)
            # for compare
            # torch.allclose(update_aggr, new_ax, atol=1e-4)
        out = self.lin_l(aggr_out)

        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, aggr_out

class SAGE(Basic):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return SAGEConvReduce(in_channels, out_channels)

class GINConvReduce(GINConv):
    def __init__(self, in_channels, out_channels, nn, aggr: Optional[Union[str, List[str], Aggregation]] = "add"):
        super().__init__(nn, aggr=aggr)
        self.in_channels = in_channels 
        self.out_channels = out_channels

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, None],
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if edge_index is None:
            aggr_out = self.update(x[0])
        else:
            aggr_out = self.propagate(edge_index, x=x)
        
        x_r = x[1]
        if x_r is not None:
            out = aggr_out + (1 + self.eps) * x_r

        return self.nn(out), aggr_out

class GIN(Basic):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=F.relu
        )
        return GINConvReduce(in_channels, out_channels, mlp)


class GCNConvReduce(GCNConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        normalize = False, 
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, normalize, aggr, **kwargs)

    # NOTE: do not add the self-loop in current implementation
    def gcn_norm(self, edge_index, num_nodes):
        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones((edge_index.size(1), ), 
                                dtype=torch.float32,
                                device=edge_index.device)
        
        deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        edge_index = torch.stack([col, row])
        return edge_index, edge_weight


    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, None],
        size = None,
    ) -> Tensor:
        if edge_index is None:
            # message
            # aggregate
            aggr_out = self.update(x)
        else:
            edge_index, edge_weight = self.gcn_norm(edge_index, x.size(0))
            coo = torch.sparse_coo_tensor(edge_index, edge_weight, 
                size=(size, x.size(0)), device=x.device)
            csr = coo.to_sparse_csr()
            aggr_out = self.propagate(csr, x=x)

        out = self.lin(aggr_out)

        return out, aggr_out


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.features = in_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConvReduce(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvReduce(hidden_channels, hidden_channels))
        self.convs.append(GCNConvReduce(hidden_channels, out_channels))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
    

    @torch.no_grad()
    def layer_wise(self, layer_i, x, edge_index, size):
        y, aggr_y = self.convs[layer_i](x, edge_index, size)
        if layer_i != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y, aggr_y
    
    @torch.no_grad()
    def layer_wise_partly(self, layer_i, new_x, x_target):
        y, _ = self.convs[layer_i](new_x, None, x_target.size(0))
        if layer_i != self.num_layers - 1:
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)
        return y
    
    def get_aggr(self, layer_i):
        return self.convs[layer_i].aggr
    
    @torch.no_grad()
    def inference(self, x, edge_index, num_nodes, batch_size, num_workers, device):
        loader = NeighborSampler(edge_index, 
                                 node_idx=torch.arange(num_nodes),
                                 sizes=[-1], 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=num_workers)
        embs = []
        aggrs = []
        for layer_i in range(self.num_layers):
            aggrs.append(torch.zeros((num_nodes, self.convs[layer_i].in_channels)))
            embs.append(torch.zeros((num_nodes, self.convs[layer_i].out_channels)))
            for step, (batch_size, ids, adjs) in enumerate(loader):
                if layer_i == 0:
                    update_x = x[ids].to(device)
                else:
                    update_x = embs[layer_i-1][ids].to(device)
                sampled_edge_index = adjs[0].to(device)
                update_y, update_a = self.layer_wise(layer_i, update_x, sampled_edge_index, batch_size)
                nodes = ids[:batch_size]
                embs[layer_i][nodes] = update_y.cpu()
                aggrs[layer_i][nodes] = update_a.cpu()

        return embs, aggrs

