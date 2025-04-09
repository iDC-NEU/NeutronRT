import torch
from lib.cpp_extension.wrapper import pmaGraph
from torch_geometric.utils import to_torch_csc_tensor, to_torch_csr_tensor

class PMAGraph:

    def __init__(
            self,
            num_nodes: int,
            edge_index: torch.Tensor,
            device: torch.device,
            max_num_edges: int = 0
    ):
        edge_index = edge_index.cpu()
        csc = to_torch_csc_tensor(edge_index)
        csr = to_torch_csr_tensor(edge_index)

        graph_csc = (csc.ccol_indices().type(torch.int64), csc.row_indices().type(torch.int64))
        graph_csr = (csr.crow_indices().type(torch.int64), csr.col_indices().type(torch.int64))

        # dst -> src
        self.graph = pmaGraph.PMAGraph(num_nodes, 
                          len(graph_csc[1]), 
                          graph_csc[0], graph_csc[1], device.index, max_num_edges)
        # src -> dst
        self.t_graph = pmaGraph.PMAGraph(num_nodes, 
                          len(graph_csr[1]), 
                          graph_csr[0], graph_csr[1], device.index, max_num_edges)
        
        # self.in_degree = degree(edge_index[1], num_nodes).to(device)
        self.in_degree = (graph_csc[0][1:] - graph_csc[0][:-1]).to(device)
        self.out_degree = (graph_csr[0][1:] - graph_csr[0][:-1]).to(device)
        self.device = device

        self.empty_int64 = torch.empty(0, dtype=torch.int64, device=device)
        self.empty_bool = torch.empty(0, dtype=torch.bool, device=device)


    def insert_edges(self, edge_index, delay=False):
        masks = torch.ones(edge_index.size(1), dtype=torch.bool)
        for step, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if self.graph.has_edge(dst, src):
                masks[step] = False
                continue
            self.graph.insert_edge(dst, src, 0)
            if not delay:
                self.t_graph.insert_edge(src, dst, 0)
            self.in_degree[dst] += 1
            self.out_degree[src] += 1
        return edge_index[:, masks]
    
    def insert_edges_actual(self, edge_index):
        for step, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            self.t_graph.insert_edge(src, dst, 0)

    def delete_edges(self, edge_index):
        masks = torch.ones(edge_index.size(1), dtype=torch.bool)
        for step, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if not self.graph.has_edge(dst, src):
                masks[step] = False
                continue
            self.graph.delete_edge(dst, src, 0)
            self.t_graph.delete_edge(src, dst, 0)
            self.in_degree[dst] -= 1
            self.out_degree[src] -= 1
        return edge_index[:, masks]

    def in_degrees(self) -> torch.Tensor:
        return self.in_degree
    
    def out_degrees(self) -> torch.Tensor:
        return self.out_degree
    
    def sample_all_nodes(self, nodes, ignore=None):
        if ignore is None:
            ignore = self.empty_bool
        return self.graph.next_layer_indices(nodes, ignore)
    

    def next_affact_nodes(self, insert_edges, delete_edges, last_nodes):
        if last_nodes is None:
            last_nodes = self.empty_int64
        src, dst, o_type = self.t_graph.next_layer(insert_edges, delete_edges, last_nodes)
        dst, reverse_indices = torch.sort(dst)
        mapped_dst, counts = torch.unique_consecutive(dst,  return_counts=True)
        src: torch.Tensor = src[reverse_indices]
        o_type: torch.Tensor = o_type[reverse_indices]
        dst_indices: torch.Tensor = torch.zeros(counts.size(0) + 1, dtype=torch.int64, device=self.device)
        torch.cumsum(counts, 0, out=dst_indices[1:])
        return src, mapped_dst, dst_indices, o_type, counts
    

    def affected_edges(self, nodes: torch.Tensor):
        src, dst = self.t_graph.affected_edges(nodes)
        torch.cuda.synchronize()
        return torch.cat([src.view(1, -1), dst.view(1, -1)], dim=0)
    

    def affected_nodes(self, nodes: torch.Tensor):
        _, dst = self.t_graph.affected_edges(nodes)
        return dst.unique()


    def next_hops(self, nodes):
        return self.t_graph.next_hops(nodes).unique()
    

    def get_next_hops(self, insert_edges, delete_edges, affect_nodes):
        if affect_nodes is None:
            affect_nodes = self.empty_int64
        src, dst, o_type = self.t_graph.next_layer(insert_edges, delete_edges, affect_nodes)
        dst, inverse_indices = torch.sort(dst)
        mapped_dst, dst_expend, counts = torch.unique_consecutive(dst, return_inverse=True, return_counts=True)
        src = src[inverse_indices]
        o_type = o_type[inverse_indices]
        dst_indices = torch.zeros(counts.size(0) + 1, dtype=torch.int64, device=self.device)
        torch.cumsum(counts, dim=0, out=dst_indices[1:])
        return src, mapped_dst, dst_indices, o_type, counts, dst_expend

    @property
    def num_edges(self):
        return self.graph.num_edges()
    
    @property
    def num_nodes(self):
        return self.graph.num_nodes()
