import torch, math
from torch_geometric.utils import to_torch_csc_tensor, to_torch_csr_tensor, add_random_edge

from typing import List

from .utils import ceil_log2, ceil_div, floor_log2, floor_div, hyper_ceil, hyper_floor

NR_STREAMS = 4


class VertexSyncElement():
    def __init__(
            self,
            length: int,
    ):
        self.degrees = torch.zeros(length, dtype=torch.int64)
        self.indices = torch.zeros(length + 1, dtype=torch.int64)

    @property
    def degree(self) -> torch.Tensor:
        return self.degrees
    
    @degree.setter
    def degree(self, value):
        self.degrees = value

    @property
    def index(self) -> torch.Tensor:
        return self.indices
    
    @index.setter
    def index(self, value):
        self.indices = value


class VertexElement():
    def __init__(
            self,
            length: int,
    ):
        self.cuda_index = torch.zeros(length, dtype=torch.int64)
        self.host_index =torch.zeros(length, dtype=torch.int64)
        self.hotness = torch.zeros((4, length), dtype=torch.uint8)
        self.valid = torch.zeros(length, dtype=torch.bool)


class PMAGraphCore():
    def __init__(
            self,
            num_nodes: int,
            indptr: torch.Tensor,
            indices: torch.Tensor,
            device: torch.device,
            max_num_edges: int = 0,
            
            # upper density thresholds
            upper_h: float = 0.75,
            upper_0: float = 1.00,

            # lower density thresholds
            lower_h: float = 0.50,
            lower_0: float = 0.25,

            # hyper parameters
            sparse_factor: float = 0.50,
            largest_empty_segment_factor: float = 1.0,
    ):
        self.num_nodes = num_nodes
        self.num_edges = len(indices)
        self.device = device

        self.upper_h = upper_h
        self.upper_0 = upper_0
        self.lower_h = lower_h
        self.lower_0 = lower_0

        self.river = 0
        self.river_lower = 0
        self.river_global = 0

        self.max_sparseness = sparse_factor / lower_0
        self.largest_empty_segment = largest_empty_segment_factor * self.max_sparseness

        self.segment_count = hyper_floor(ceil_div(self.num_nodes, ceil_log2(self.num_nodes)))
        self.segment_size = ceil_div(self.num_nodes, self.segment_count)
        
        self.tree_height = floor_log2(self.segment_count)
        self.delta_upper = (upper_0 - upper_h) / self.tree_height
        self.delta_lower = (lower_h - lower_0) / self.tree_height

        use_num_edges = max_num_edges if max_num_edges > 0 else self.num_edges
        self.edge_capacity = int(use_num_edges * self.max_sparseness)

        self.streams: List[torch.Stream] = []
        for _ in range(NR_STREAMS):
            self.streams.append(torch.Stream(device))

        self.bind(
            indptr=indptr,
            indices=indices,
        )

    
    def avg_degree(self):
        return self.num_edges // self.num_nodes
    

    def segment_id_of(self, vertices: torch.Tensor):
        return (vertices // self.segment_size) + self.segment_count
    

    def update_segment_edge_total(self, vertices: torch.Tensor, counts: torch.Tensor):
        ids = self.segment_id_of(vertices)
        while (ids > 0).any():
            self.segment_edges_total.index_add_(0, ids, counts)
            ids = ids // 2

        # clear 0 index element
        self.segment_edges_total[0] = 0


    def has_edge(self, edge_index: torch.Tensor):
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        src, dst = edge_index
        indices = self.relate_indices(src)

        for i, (index, target) in enumerate(zip(indices, dst)):
            begin, end = index
            mask[i] = (self.edges[begin:end] == target).any()
        return mask


    def bind(
            self,
            indptr: torch.Tensor,
            indices: torch.Tensor,
    ):
        # allocate memory
        self.segment_edges_actual = torch.zeros((self.segment_count * 2), dtype=torch.int64)
        self.segment_edges_total = torch.zeros((self.segment_count * 2), dtype=torch.int64)
        self.sync_vertices = VertexSyncElement(self.num_nodes)
        self.edges = torch.ones(self.edge_capacity * 2, dtype=torch.int64).neg_()

        self.edges[0:self.num_edges] = indices
        self.sync_vertices.degree = torch.diff(indptr)
        self.sync_vertices.index = indptr

        mask = self.sync_vertices.degree > 0
        node_idx = torch.arange(self.num_nodes)
        node_chosen = node_idx[mask]
        deg_chosen = self.sync_vertices.degree[mask]
        t_seg_id = node_chosen // self.segment_size
        it = t_seg_id + self.segment_count

        while (it > 0).any():
            self.segment_edges_actual.scatter_add_(
                dim=0,
                index=it,
                src=deg_chosen,
            )
            it = it // 2

        # clear 0 index element
        self.segment_edges_actual[0] = 0

        node_fix = node_idx[~mask]
        for i in node_fix:
            self.sync_vertices.index[i] = self.sync_vertices.index[i - 1] + self.sync_vertices.degree[i - 1]
        
        self.spread_weight_v1(0, self.num_nodes)


    def spread_weight_v1(self, vertex_begin: int, vertex_end: int):
        assert vertex_begin == 0, f'expect begin to be 0 but get {vertex_begin} instead'
        gaps = self.edge_capacity - self.num_edges
        new_pos = self.calculate_position_v1(vertex_begin, vertex_end, gaps, self.num_edges)


        cur_degrees = self.sync_vertices.degree[vertex_begin+1:vertex_end]
        rd_index = self.sync_vertices.index[vertex_begin+1:vertex_end] + cur_degrees - 1
        wr_index = new_pos[vertex_begin+1:vertex_end] + cur_degrees - 1
        assert (wr_index >= rd_index).all(), 'index anomaly occurred while spreading elements'

        for i in range(cur_degrees.max()):
            mask = cur_degrees > i
            self.edges[wr_index[mask]] = self.edges[rd_index[mask]]
            rd_index[mask] -= 1
            wr_index[mask] -= 1

        self.sync_vertices.index[vertex_begin+1:vertex_end] = new_pos[vertex_begin+1:vertex_end]

        del new_pos
        self.recount_segment_total()


    def recount_segment_total(self):
        self.segment_edges_total.zero_()

        next_indices = torch.minimum(
            torch.tensor(self.num_nodes - 1),
            torch.arange(1, self.segment_count + 1) * self.segment_size
        )
        curr_indices = torch.minimum(
            torch.tensor(self.num_nodes - 1),
            torch.arange(self.segment_count) * self.segment_size
        )
        next_starters = torch.where(
            torch.arange(self.segment_count) == self.segment_count - 1,
            self.edge_capacity,
            self.sync_vertices.index[next_indices]
        )

        segment_totals = next_starters - self.sync_vertices.index[curr_indices]

        ids = torch.arange(self.segment_count) + self.segment_count
        while (ids > 0).any():
            self.segment_edges_total.index_add_(0, ids, segment_totals)
            ids = ids // 2

        # clear 0 index element
        self.segment_edges_total[0] = 0


    def calculate_position_v1(
            self,
            vertex_begin: int,
            vertex_end: int,
            gaps: int,
            total_degree: int,
    ):
        size = vertex_end - vertex_begin
        total_degree += size

        index = torch.tensor([self.sync_vertices.index[vertex_begin]], dtype=torch.float64)
        step = gaps / total_degree

        min_degrees = self.sync_vertices.degree[vertex_begin:vertex_end]
        min_degrees[min_degrees == 0] = 1
        min_degrees = self.sync_vertices.degree[vertex_begin:vertex_end] + step * min_degrees
        index = torch.cat([index, min_degrees])
        index.cumsum_(dim=0)
        return index[:-1].to(dtype=torch.int64)
    

    def rebalance_wrapper(self, vertices: torch.Tensor):
        segment_id = self.segment_id_of(vertices)
        while segment_id > 0:
            segment_start = (segment_id - self.segment_count) * self.segment_size
            segment_end = min(segment_start + self.segment_size, self.num_nodes)
            self.rebalance_weighted(segment_start, segment_end, segment_id)
            segment_id //= 2


    def rebalance_weighted(self, start_vertex: int, end_vertex: int, pma_idx: int):
        gaps = self.edge_capacity - self.num_edges
        new_pos = self.calculate_position_v1(start_vertex, end_vertex, gaps, self.num_edges)

        cur_degrees = self.sync_vertices.degree[start_vertex:end_vertex]
        rd_indices = self.sync_vertices.index[start_vertex:end_vertex] + cur_degrees - 1
        wr_indices = new_pos[start_vertex:end_vertex] + cur_degrees - 1

        assert (wr_indices >= rd_indices).all(), 'index anomaly occurred while rebalancing elements'

        max_degree = cur_degrees.max().item()
        for i in range(max_degree):
            mask = cur_degrees > i
            self.edges[wr_indices[mask]] = self.edges[rd_indices[mask]]
            rd_indices[mask] -= 1
            wr_indices[mask] -= 1

        self.sync_vertices.index[start_vertex:end_vertex] = new_pos[start_vertex:end_vertex]
        del new_pos
        self.recount_segment_total()

    
    def insert_edges(
            self,
            edge_index: torch.Tensor,
    ) -> torch.Tensor:
        self.num_edges += edge_index.size(1)
        src, dst = edge_index
        segidx = self.segment_id_of(src)
        locs = self.sync_vertices.index[src] + self.sync_vertices.degree[src]

        for s, d, cur_segment, loc in zip(src, dst, segidx, locs):
            right_free_slot = -1
            left_free_slot = -1
            left_vertex = s
            right_vertex = s

            left_vertex_boundary = (s // self.segment_size) * self.segment_size
            right_vertex_boundary = min(left_vertex_boundary + self.segment_size, self.num_nodes - 1)

            for i in range(s, right_vertex_boundary):
                if self.sync_vertices.index[i + 1] > (self.sync_vertices.index[i] + self.sync_vertices.degree[i]):
                    right_free_slot = self.sync_vertices.index[i] + self.sync_vertices.degree[i]
                    right_vertex = i
                    break

            if right_free_slot == -1 and right_vertex_boundary == (self.num_nodes - 1) and self.edge_capacity > (1 + self.sync_vertices.index[self.num_nodes - 1] + self.sync_vertices.degree[self.num_nodes - 1]):
                right_free_slot = self.sync_vertices.index[self.num_nodes - 1] + self.sync_vertices.degree[self.num_nodes - 1]
                right_vertex = self.num_nodes - 1

            if right_free_slot == -1:
                for i in range(s, left_vertex_boundary, -1):
                    if self.sync_vertices.index[i] > (self.sync_vertices.index[i - 1] + self.sync_vertices.degree[i - 1]):
                        left_free_slot = self.sync_vertices.index[i] - 1
                        left_vertex = i
                        break

            if right_free_slot != -1:
                if right_free_slot >= loc:
                    self.edges[loc + 1:right_free_slot + 1] = self.edges[loc:right_free_slot]
                    self.sync_vertices.index[s + 1:right_vertex + 1] += 1
                    if cur_segment != self.segment_id_of(right_vertex):
                        self.update_segment_edge_total(s, 1)
                        self.update_segment_edge_total(right_vertex, -1)
                self.edges[loc] = d
                self.sync_vertices.degree[s] += 1
            elif left_free_slot != -1:
                if left_free_slot < loc:
                    self.edges[left_free_slot:loc - 1] = self.edges[left_free_slot + 1:loc]
                    self.sync_vertices.index[left_vertex:s + 1] -= 1
                    if cur_segment != self.segment_id_of(left_vertex):
                        self.update_segment_edge_total(s, 1)
                        self.update_segment_edge_total(left_vertex, -1)
                self.edges[loc - 1] = d
                self.sync_vertices.degree[s] += 1
            else:
                raise RuntimeError("should not happen")

            j = cur_segment
            while j > 0:
                self.segment_edges_actual[j] += 1
                j //= 2

            self.rebalance_wrapper(s)


    def delete_edges(
            self,
            edge_index: torch.Tensor,
    ) -> torch.Tensor:
        self.num_edges -= edge_index.size(1)
        src, dst = edge_index
        segidx = self.segment_id_of(src)
        lefts = self.sync_vertices.index[src]
        rights = self.sync_vertices.index[src] + self.sync_vertices.degree[src]

        for s, d, cur_segment, l, r in zip(src, dst, segidx, lefts, rights):
            for i in range(l, r):
                if self.edges[i] == d:
                    self.edges[i:r - 1] = self.edges[i + 1:r]
                    self.sync_vertices.degree[s] -= 1
                    self.edges[r - 1] = -1
                    break

            j = cur_segment
            while j > 0:
                self.segment_edges_actual[j] -= 1
                j //= 2


    def relate_indices(
            self,
            targets: torch.Tensor,
    ):
        begin = self.sync_vertices.index[targets]
        end = self.sync_vertices.index[targets + 1]
        return torch.cat([begin, end]).view(2, -1).transpose_(0, 1)


    def degree(self):
        return self.sync_vertices.degree


class PMAGraph():
    def __init__(
            self,
            num_nodes: int,
            edge_index: torch.Tensor,
            device: torch.device,
            max_num_edges: int = 0
    ):
        self.num_nodes = num_nodes
        self.device = device
        self.max_num_edges = max_num_edges


        csc: torch.Tensor = to_torch_csc_tensor(edge_index)
        # dst -> src
        self.graph = PMAGraphCore(
            num_nodes=num_nodes,
            indptr=csc.ccol_indices().type(torch.int64),
            indices=csc.row_indices().type(torch.int64),
            device=device,
            max_num_edges=max_num_edges,
        )

        csr: torch.Tensor = to_torch_csr_tensor(edge_index)
        # src -> dst
        self.t_graph = PMAGraphCore(
            num_nodes=num_nodes,
            indptr=csr.crow_indices().type(torch.int64),
            indices=csr.col_indices().type(torch.int64),
            device=device,
            max_num_edges=max_num_edges,
        )


    def insert_edges(
            self,
            edge_index: torch.Tensor,
            delay: bool = False
    ) -> torch.Tensor:
        mask = ~self.graph.has_edge(edge_index)
        self.graph.insert_edges(edge_index[:, mask])
        if not delay:
            self.t_graph.insert_edges(edge_index[:, mask])
        return edge_index[:, mask]


    def delete_edges(
            self,
            edge_index: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.graph.has_edge(edge_index)
        self.graph.delete_edges(edge_index[:, mask])
        self.t_graph.delete_edges(edge_index[:, mask])
        return edge_index[:, mask]
        

    def insert_edges_actual(
            self,
            edge_index: torch.Tensor,
    ):
        self.t_graph.insert_edges(edge_index)


    def in_degrees(self) -> torch.Tensor:
        return self.graph.degree()


    def relate_in_edges(
            self,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.graph.relate_indices(targets)


    def out_degrees(self) -> torch.Tensor:
        return self.t_graph.degree()


    def relate_out_edges(
            self,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.t_graph.relate_indices(targets)