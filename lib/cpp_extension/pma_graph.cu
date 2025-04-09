#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "pma_graph.cu.hpp"
#include "reindex.cu.hpp"
#include "cuda_random.cu.hpp"
#include "functor.cu.hpp"
#include "utils.h"

template <typename IdType>
HostOrderedHashTable<IdType> *
FillWithDuplicates(const IdType *const input, const size_t num_input,
                   cudaStream_t stream,
                   thrust::device_vector<IdType> &unique_items)
{
    const auto policy = thrust::cuda::par.on(stream);
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    auto host_table = new HostOrderedHashTable<IdType>(num_input, 1);
    DeviceOrderedHashTable<IdType> device_table = host_table->DeviceHandle();

    generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>
        <<<grid, block, 0, stream>>>(input, num_input, device_table);
    thrust::device_vector<int> item_prefix(num_input + 1, 0);

    using it = thrust::counting_iterator<IdType>;
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    thrust::for_each(it(0), it(num_input),
                     [count = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table,
                      in = input] __device__(IdType i) mutable
                     {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i)
                         {
                             count[i] = 1;
                         }
                     });
    thrust::exclusive_scan(item_prefix.begin(), item_prefix.end(),
                           item_prefix.begin());
    size_t tot = item_prefix[num_input];
    unique_items.resize(tot);

    thrust::for_each(it(0), it(num_input),
                     [prefix = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table, in = input,
                      u = thrust::raw_pointer_cast(unique_items.data())] __device__(IdType i) mutable
                     {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i)
                         {
                             mapping.local = prefix[i];
                             u[prefix[i]] = in[i];
                         }
                     });
    return host_table;
}

void PMAGraph::Bind(int64_t nnodes, int64_t nedges,
                    int64_t *row_start, int64_t *edge_dst, 
                    int64_t indicated_size)
//   int64_t *edge_weights, int64_t *node_weights)
{
    this->nnodes = nnodes;
    this->nedges = nedges;

    // this->edge_weights = edge_weights;
    // this->node_weights = node_weights;

    compute_capacity(indicated_size);
    this->segment_edges_actual = (int64_t *)malloc(sizeof(int64_t) * segment_count * 2);
    this->segment_edges_total = (int64_t *)malloc(sizeof(int64_t) * segment_count * 2);
    memset(this->segment_edges_actual, 0, sizeof(int64_t) * this->segment_count * 2);
    memset(this->segment_edges_total, 0, sizeof(int64_t) * this->segment_count * 2);
    tree_height = floor_log2(segment_count);
    delta_up = (up_0 - up_h) / tree_height;
    delta_low = (low_h - low_0) / tree_height;

    cudaMallocHost((void **)(&(this->edges_)), sizeof(int64_t) * (this->elem_capacity_max));
    cudaMallocHost((void **)(&(this->sync_vertices_)), (nnodes + 1) * sizeof(vertex_sync_element_t));

    memset(edges_, -1, sizeof(int64_t) * elem_capacity_max);
    memset(sync_vertices_, 0, sizeof(vertex_sync_element_t) * (nnodes + 1));

    for (int64_t node = 0; node < nnodes; node++)
    {
        int64_t begin_edge = row_start[node];
        int64_t end_edge = row_start[node + 1];
        sync_vertices_[node].degree = end_edge - begin_edge;
                sync_vertices_[node].index = begin_edge;
        
        // for (int64_t edge = begin_edge; edge < end_edge; edge++)
        if (begin_edge < end_edge)
        {
            int32_t t_segment_id = node / segment_size;
            int32_t j = t_segment_id + segment_count;
            // printf("t_src = %d, j = %d, t_segment_id = %d\n",node,j,t_segment_id);
            while (j > 0)
            {
                segment_edges_actual[j] += (end_edge - begin_edge);
                j /= 2;
            }
        }
    }

    for (int64_t e = 0; e < nedges; e++)
    {
        edges_[e] = edge_dst[e];
    }

    // std::cout << " look edge_dst[nedges-1] : " << edge_dst[nedges - 1]
    //           << " look edge_dst[nedges-2] : " << edge_dst[nedges - 2]
    //           << " nedges-1: " << (nedges - 1) << " nedges-2: " << (nedges - 2) << std::endl;
    // std::cout << " edge_[nedges-1] : " << edges_[nedges - 1]
    //           << " look edge_dst[nedges-2] : " << edges_[nedges - 2]
    //           << " nedges-1: " << (nedges - 1) << " nedges-2: " << (nedges - 2) << std::endl;

    for (int32_t i = 1; i < nnodes; i++)
    {
        if (sync_vertices_[i].degree == 0)
    sync_vertices_[i].index = sync_vertices_[i - 1].index + sync_vertices_[i - 1].degree;
    }
    spread_weighted_V1(0, nnodes);
    // if(DEBUG) std::cout << ">>>>>>>> root-density: " << ((double) segment_edges_actual[1] / (double) segment_edges_total[1]) << std::endl;
}

/*
    void PrintHistogram(int32_t *&p_in_degree, int32_t *&p_out_degree)
    {
        int32_t *in_degree = new int32_t[nnodes];
        int32_t *out_degree = new int32_t[nnodes];
        for (int64_t node = 0; node < nnodes; node++)
        {
            int64_t begin_edge = this->sync_vertices_[node].index;
            // int64_t begin_edge = this->begin_edge(node),
            // end_edge = this->end_edge(node);
            // printf("node %d,begin_edge %d, end_edge%d\n",node,begin_edge,end_edge);

            // out_degree[node] = end_edge - begin_edge;
            out_degree[node] = this->sync_vertices_[node].degree;
            for (int64_t edge = begin_edge; edge < begin_edge + out_degree[node]; edge++)
            {
                int64_t dest = this->edge_dest(edge);
                // if(dest != -1){
                in_degree[dest]++;
                // }
            }
        }
        GROUTE_CUDA_CHECK(cudaMalloc(&p_in_degree, nnodes * sizeof(int64_t)));
        GROUTE_CUDA_CHECK(
            cudaMemcpy(p_in_degree, in_degree, nnodes * sizeof(int64_t),
                       cudaMemcpyHostToDevice));

        GROUTE_CUDA_CHECK(cudaMalloc(&p_out_degree, nnodes * sizeof(int64_t)));
        GROUTE_CUDA_CHECK(
            cudaMemcpy(p_out_degree, out_degree, nnodes * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
    }

    void PrintHistogram_update(int32_t *&p_in_degree, int32_t *&p_out_degree)
    {
        int32_t *in_degree = new int32_t[nnodes];
        int32_t *out_degree = new int32_t[nnodes];
        for (int64_t node = 0; node < nnodes; node++)
        {
            int64_t begin_edge = this->sync_vertices_[node].index;
            // int64_t begin_edge = this->begin_edge(node),
            // end_edge = this->end_edge(node);
            // printf("node %d,begin_edge %d, end_edge%d\n",node,begin_edge,end_edge);

            // out_degree[node] = end_edge - begin_edge;
            out_degree[node] = sync_vertices_[node].degree;
            for (int64_t edge = begin_edge; edge < (begin_edge + out_degree[node]); edge++)
            {
                int64_t dest = this->edge_dest(edge);
                // if(dest != -1){
                in_degree[dest]++;
                // }
            }
        }
        // GROUTE_CUDA_CHECK(cudaMalloc(&p_in_degree, nnodes * sizeof(int64_t)));
        GROUTE_CUDA_CHECK(
            cudaMemcpy(p_in_degree, in_degree, nnodes * sizeof(int64_t),
                       cudaMemcpyHostToDevice));

        // GROUTE_CUDA_CHECK(cudaMalloc(&p_out_degree, nnodes * sizeof(int64_t)));
        GROUTE_CUDA_CHECK(
            cudaMemcpy(p_out_degree, out_degree, nnodes * sizeof(int64_t),
                       cudaMemcpyHostToDevice));
    }
*/

void PMAGraph::insert_edge(int64_t src, int64_t dst, int64_t value)
{
    nedges += 1;
    int64_t current_segment = get_segment_id(src);
    // raqib: we now allow to look for space beyond the segment boundary
    // assert(segment_edges_total[current_segment] > segment_edges_actual[current_segment] && "There is no enough space in the current segment for performing this insert!");

    int64_t loc = sync_vertices_[src].index + sync_vertices_[src].degree;
    int64_t right_free_slot = -1;
    int64_t left_free_slot = -1;
    int64_t left_vertex = src, right_vertex = src;
    int64_t left_vertex_boundary, right_vertex_boundary;

    if (segment_edges_total[current_segment] > segment_edges_actual[current_segment])
    {
        left_vertex_boundary = (src / segment_size) * segment_size;
        right_vertex_boundary = min(left_vertex_boundary + segment_size, nnodes - 1);
    }
    else
    {
        int64_t curr_seg_size = segment_size, j = current_segment;
        while (j)
        {
            if (segment_edges_total[j] > segment_edges_actual[j])
                break;
            j /= 2;
            curr_seg_size *= 2;
        }
        left_vertex_boundary = (src / curr_seg_size) * curr_seg_size;
        right_vertex_boundary = min(left_vertex_boundary + curr_seg_size, nnodes - 1);
    }

    // search right side for a free slot
    // raqib: shouldn't we search within the pma leaf?
    for (int64_t i = src; i < right_vertex_boundary; i++)
    {
        if (sync_vertices_[i + 1].index > (sync_vertices_[i].index + sync_vertices_[i].degree))
        {
            right_free_slot = sync_vertices_[i].index + sync_vertices_[i].degree; // we get a free slot here
            right_vertex = i;
            break;
        }
    }

    // in the last segment where we skipped the last vertex
    // if (right_free_slot == -1 && elem_capacity > (1 + vertices_[nnodes - 1].index + vertices_[nnodes - 1].degree))
    if (right_free_slot == -1 && right_vertex_boundary == (nnodes - 1) && elem_capacity > (1 + sync_vertices_[nnodes - 1].index + sync_vertices_[nnodes - 1].degree))
    {
        right_free_slot = sync_vertices_[nnodes - 1].index + sync_vertices_[nnodes - 1].degree;
        right_vertex = nnodes - 1;
    }

    // if no space on the right side, search the left side
    if (right_free_slot == -1)
    {
        for (int64_t i = src; i > left_vertex_boundary; i--)
        {
            if (sync_vertices_[i].index > (sync_vertices_[i - 1].index + sync_vertices_[i - 1].degree))
            {
                left_free_slot = sync_vertices_[i].index - 1; // we get a free slot here
                left_vertex = i;
                break;
            }
        }
    }

    // if(DEBUG) cout << "left_free_slot: " << left_free_slot << ", right_free_slot: " << right_free_slot << endl;
    // if(DEBUG) cout << "left_free_vertex: " << left_vertex << ", right_free_vertex: " << right_vertex << endl;

    // found free slot on the right
    if (right_free_slot != -1)
    {
        if (right_free_slot >= loc) // move elements to the right to get the free slot
        {
            for (int64_t i = right_free_slot; i > loc; i--)
            {
                edges_[i] = edges_[i - 1];
                // edges_[i].t = edges_[i-1].t;
            }

            for (int64_t i = src + 1; i <= right_vertex; i++)
            {
                sync_vertices_[i].index += 1;
            }

            // update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
            if (current_segment != get_segment_id(right_vertex))
            {
                update_segment_edge_total(src, 1);
                update_segment_edge_total(right_vertex, -1);
            }
        }
        edges_[loc] = dst;

        sync_vertices_[src].degree += 1;
    }
    else if (left_free_slot != -1)
    {
        if (left_free_slot < loc)
        {
            for (int64_t i = left_free_slot; i < loc - 1; i++)
            {
                edges_[i] = edges_[i + 1];
            }

            for (int64_t i = left_vertex; i <= src; i++)
            {
                sync_vertices_[i].index -= 1;
            }

            // update the segment_edges_total for the source-vertex and right-vertex's segment if it lies in different segment
            if (current_segment != get_segment_id(left_vertex))
            {
                update_segment_edge_total(src, 1);
                update_segment_edge_total(left_vertex, -1);
            }
        }
        edges_[loc - 1] = dst;

        sync_vertices_[src].degree += 1;
    }
    else
    {
        assert(false && "Should not happen");
    }

    // we insert a new edge, increasing the degree of the whole subtree
    int64_t j = current_segment;
    // if(debug) cout << "segment where insert happened: " << j << endl;
    while (j > 0)
    {
        segment_edges_actual[j] += 1;
        j /= 2;
    }

    // whether we need to update segment_edges_total
    // raqib: I think we should do the re-balance first and then make the insertion.
    // This will always ensure the empty space for the insertion.

    // num_edges_ += 1;

    rebalance_wrapper(src);
    // todo: we should also consider rebalancing the donner vertex's segment
}

void PMAGraph::delete_edge(int64_t src, int64_t dst, int64_t weight)
{
    nedges -= 1;

    int64_t current_segment = get_segment_id(src);

    int64_t left = sync_vertices_[src].index;
    int64_t right = sync_vertices_[src].index + sync_vertices_[src].degree;
    // printf("[left %u, right %u)] Called!\n", left, right);
    for (int64_t i = left; i < right; i++)
    {
        if (edges_[i] == dst)
        {

            for (int64_t j = (i + 1); j < right; j++)
            {
                edges_[j - 1] = edges_[j];
            }

            sync_vertices_[src].degree -= 1;

            edges_[right - 1] = -1;

            // std::sort(vertices_[src].index, (vertices_[src].index+vertices_[src].degree));
            break;
        }
    }

    int64_t j = current_segment;
    // if(debug) cout << "segment where insert happened: " << j << endl;
    while (j > 0)
    {
        segment_edges_actual[j] -= 1;
        j /= 2;
    }

}

void PMAGraph::compute_capacity(int64_t indicated_size)
{
    int64_t &nnodes = this->nnodes;
    auto &segment_size = this->segment_size;
    auto &segment_count = this->segment_count;
    auto &nedges = this->nedges;
    auto &elem_capacity = this->elem_capacity;
    auto &max_sparseness = this->max_sparseness;

    segment_size = ceil_log2(nnodes);
    segment_count = ceil_div(nnodes, segment_size); // Ideal number of segments
    // The number of segments has to be a power of 2, though.
    segment_count = hyperfloor(segment_count);
    segment_size = ceil_div(nnodes, segment_count);
    // printf("real nnodes = %d\n",nnodes);
    // nnodes = segment_count * segment_size;
    elem_capacity = nedges * max_sparseness;
    if (indicated_size > 0)
        elem_capacity = indicated_size * max_sparseness;
    auto &elem_capacity_max = this->elem_capacity_max;
    elem_capacity_max = 2 * elem_capacity;
}

void PMAGraph::spread_weighted_V1(int32_t start_vertex, int32_t end_vertex)
{
    assert(start_vertex == 0 && "start-vertex is expected to be 0 here.");
    int64_t gaps = elem_capacity - nedges;
    int64_t *new_positions = calculate_positions_V1(start_vertex, end_vertex, gaps, nedges);

    // if(DEBUG){
    //     for(int32_t curr_vertex = start_vertex;  curr_vertex < end_vertex; curr_vertex++){
    //         std::cout << "vertex-id: " << curr_vertex << ", index: " << sync_vertices_[curr_vertex].index;
    //         std::cout << ", degree: " << sync_vertices_[curr_vertex].degree << ", new position: " << new_positions[curr_vertex] << std::endl;
    //     }
    // }

    int64_t read_index, write_index, curr_degree;
    // int64_t read_index_w, write_index_w, curr_degree_w;
    for (int32_t curr_vertex = end_vertex - 1; curr_vertex > start_vertex; curr_vertex -= 1)
    {
        curr_degree = sync_vertices_[curr_vertex].degree;
        read_index = sync_vertices_[curr_vertex].index + curr_degree - 1;
        write_index = new_positions[curr_vertex] + curr_degree - 1;

        // if(write_index < read_index) {
        //     std::cout << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << std::endl;
        // }
        assert(write_index >= read_index && "index anomaly occurred while spreading elements");

        for (int32_t i = 0; i < curr_degree; i++)
        {
            // if (curr_vertex == 105153950 || curr_vertex == 105153951)
            // {
            //     std::cout << " 前面 "
            //               << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: " << edges_[read_index] << " 写到的边： " << edges_[write_index] << std::endl;
            // }
            edges_[write_index] = edges_[read_index];
            // if (curr_vertex == 105153950 || curr_vertex == 105153951)
            // {
            //     std::cout << " 后面 "
            //               << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: " << edges_[read_index] << " 写到的边： " << edges_[write_index] << std::endl;
            // }
            if (read_index < new_positions[curr_vertex])
                edges_[read_index] = -1;
            // if (curr_vertex == 105153950 || curr_vertex == 105153951)
            // {
            //     std::cout << " 最后面为-1 "
            //               << "current-vertex: " << curr_vertex << ", read: " << read_index << ", write: " << write_index << ", degree: " << curr_degree << " 读到的边: " << edges_[read_index] << " 写到的边： " << edges_[write_index] << std::endl;
            // }
            write_index--;
            read_index--;
        }

        sync_vertices_[curr_vertex].index = new_positions[curr_vertex];
    }
    free(new_positions);
    new_positions = nullptr;
    recount_segment_total();
}

void PMAGraph::recount_segment_total()
{
    // count the size of each segment in the tree
    memset(segment_edges_total, 0, sizeof(int64_t) * segment_count * 2);
    for (int64_t i = 0; i < segment_count; i++)
    {
        int64_t next_index = (i + 1) * segment_size >= this->nnodes ? this->nnodes : (i + 1) * segment_size;
        int64_t curr_index = i * segment_size >= this->nnodes ? this->nnodes : i * segment_size;
        int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : sync_vertices_[next_index].index;
        int64_t segment_total_p = next_starter - sync_vertices_[curr_index].index;
        int32_t j = i + segment_count; // tree leaves
        while (j > 0)
        {
            segment_edges_total[j] += segment_total_p;
            j /= 2;
        }
    }
}

void PMAGraph::recount_segment_total(int64_t start_vertex, int64_t end_vertex)
{
    int64_t start_seg = get_segment_id(start_vertex) - segment_count;
    int64_t end_seg = get_segment_id(end_vertex) - segment_count;

    for (int64_t i = start_seg; i < end_seg; i += 1)
    {
        int64_t next_index = (i + 1) * segment_size >= this->nnodes ? this->nnodes : (i + 1) * segment_size;
        int64_t curr_index = i * segment_size >= this->nnodes ? this->nnodes : i * segment_size;
        int64_t next_starter = (i == (segment_count - 1)) ? (elem_capacity) : sync_vertices_[next_index].index;
        int64_t segment_total_p = next_starter - sync_vertices_[curr_index].index;
        int64_t j = i + segment_count; // tree leaves
        segment_total_p -= segment_edges_total[j];
        while (j > 0)
        {
            segment_edges_total[j] += segment_total_p;
            j /= 2;
        }
    }
}

int64_t *PMAGraph::calculate_positions_V1(int64_t start_vertex, int64_t end_vertex, int64_t gaps, int64_t total_degree)
{
    int64_t size = end_vertex - start_vertex;
    int64_t *new_index = (int64_t *)calloc(size, sizeof(int64_t));
    total_degree += size;

    double index_d = sync_vertices_[start_vertex].index;
    double step = ((double)gaps) / total_degree; // per-edge step
    for (int32_t i = start_vertex; i < end_vertex; i++)
    {
        new_index[i - start_vertex] = index_d;

        if (i > start_vertex)
        {
            if (new_index[i - start_vertex] < new_index[(i - 1) - start_vertex] + sync_vertices_[i - 1].degree) {
                printf("new_index %d %d %lld %lld %lld %f %lld %lld\n", i, start_vertex, new_index[i - start_vertex], new_index[(i - 1) - start_vertex], sync_vertices_[i - 1].degree, step, gaps, total_degree);
            }
            assert(new_index[i - start_vertex] >= new_index[(i - 1) - start_vertex] + sync_vertices_[i - 1].degree && "Edge-list can not be overlapped with the neighboring vertex!");
        }
        int64_t min_degree = sync_vertices_[i].degree;
        if (min_degree == 0)
            min_degree = 1;
        index_d += (sync_vertices_[i].degree + step * min_degree);
    }
    return new_index;
}

void PMAGraph::rebalance_wrapper(int64_t src)
{
    // printf("[rebalance_wrapper] Called\n");
    int64_t height = 0;
    int64_t window = (src / segment_size) + segment_count;
    double density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
    double up_height = up_0 - (height * delta_up);
    double low_height = low_0 + (height * delta_low);

    // std::cout << "Window: " << window << ", density: " << density << ", up_height: " << up_height << ", low_height: " << low_height << std::endl;

    // while (window > 0 && (density < low_height || density >= up_height))
    while (window > 0 && (density >= up_height))
    {
        // Repeatedly check window containing an increasing amount of segments
        // Now that we recorded the number of elements and occupancy in segment_edges_total and segment_edges_actual respectively;
        // so, we are going to check if the current window can fulfill the density thresholds.
        // density = gap / segment-size

        // Go one level up in our conceptual PMA tree
        window /= 2;
        height += 1;

        up_height = up_0 - (height * delta_up);
        low_height = low_0 + (height * delta_low);

        density = (double)(segment_edges_actual[window]) / (double)segment_edges_total[window];
        // std::cout << "Window: " << window << ", density: " << density << ", up_height: " << up_height << ", low_height: " << low_height << std::endl;
    }

    if (!height)
    {
        // rebalance is not required in the single pma leaf
        return;
    }
    int64_t left_index, right_index;
    // if (density >= low_height && density < up_height)
    if (density < up_height)
    {
        // Found a window within threshold
        int64_t window_size = segment_size * (1 << height);
        left_index = (src / window_size) * window_size;
        right_index = std::min(left_index + window_size, nnodes);

        // do degree-based distribution of gaps
        rebalance_weighted(left_index, right_index, window);
    }
    else
    {
        // Rebalance not possible without increasing the underlying array size.
        // need to resize the size of "edges_" array
        resize();
    }
}

void PMAGraph::rebalance_weighted(int64_t start_vertex, int64_t end_vertex, int64_t pma_idx)
{    
    // assert(to > from && "Invalid range found while doing weighted rebalance");
    // int64_t capacity = to - from;
    // assert(segment_edges_total[pma_idx] == capacity && "Segment capacity is not matched with segment_edges_total");
    int64_t gaps = segment_edges_total[pma_idx] - segment_edges_actual[pma_idx];
    int64_t size = end_vertex - start_vertex;
    int64_t *new_index = calculate_positions_V1(start_vertex, end_vertex, gaps, segment_edges_actual[pma_idx]);
    int64_t index_boundary = (end_vertex >= nnodes) ? elem_capacity : sync_vertices_[end_vertex].index;
    assert(new_index[size - 1] + sync_vertices_[end_vertex - 1].degree <= elem_capacity && "out of elem_capacity!");
    assert(new_index[size - 1] + sync_vertices_[end_vertex - 1].degree <= index_boundary && "Rebalance (weighted) index calculation is wrong!");
    int64_t ii, jj;
    int64_t curr_vertex = start_vertex + 1, next_to_start;
    int64_t read_index, last_read_index, write_index;

    int64_t from = sync_vertices_[start_vertex].index;
    int64_t to = ((end_vertex - 1) >= nnodes) ? elem_capacity : (sync_vertices_[end_vertex - 1].index + sync_vertices_[end_vertex - 1].degree);
    // int64_t *test_index = (int64_t *)malloc(sizeof(int64_t) * size);
    // int64_t *test_edges = (int64_t *)malloc(sizeof(int64_t) * index_boundary);
    int64_t *copy_edges = (int64_t *)malloc(sizeof(int64_t) * (to - from));
    memcpy((void*)copy_edges, (void*)(&edges_[from]), (to - from) * sizeof(int64_t));
    for (int64_t curr_i = start_vertex; curr_i < end_vertex; curr_i++)
    {
        if (new_index[curr_i - start_vertex] != sync_vertices_[curr_i].index) {
            int64_t node_index = sync_vertices_[curr_i].index;
            sync_vertices_[curr_i].index = new_index[curr_i - start_vertex];
            memcpy((void*)(&edges_[sync_vertices_[curr_i].index]), 
                   (void*)(&copy_edges[node_index - from]), 
                   sync_vertices_[curr_i].degree * sizeof(int64_t));
        }
    }

    /*while (curr_vertex < end_vertex)
    {
        for (ii = curr_vertex; ii < end_vertex; ii++)
        {
            if (new_index[ii - start_vertex] <= sync_vertices_[ii].index)
                break;
        }
        if (ii == end_vertex)
            ii -= 1;
        // assert(new_index[ii - start_vertex] <= sync_vertices_[ii].index && "This should not happen!");
        next_to_start = ii + 1;
        if (new_index[ii - start_vertex] <= sync_vertices_[ii].index)
        {
            // now it is guaranteed that, ii's new-starting index is less than or equal to it's old-starting index
            jj = ii;
            read_index = sync_vertices_[jj].index;
            last_read_index = read_index + sync_vertices_[jj].degree;
            write_index = new_index[jj - start_vertex];

            while (read_index < last_read_index)
            {
                edges_[write_index] = edges_[read_index];
                write_index++;
                read_index++;
            }
            // update the index to the new position
            sync_vertices_[jj].index = new_index[jj - start_vertex];

            ii -= 1;
        }

        // from current_vertex to ii, the new-starting index is greater than to it's old-starting index
        for (jj = ii; jj >= curr_vertex; jj -= 1)
        {
            read_index = sync_vertices_[jj].index + sync_vertices_[jj].degree - 1;
            last_read_index = sync_vertices_[jj].index;
            write_index = new_index[jj - start_vertex] + sync_vertices_[jj].degree - 1;

            while (read_index >= last_read_index)
            {
                edges_[write_index] = edges_[read_index];
                // weights_[write_index] = weights_[read_index];
                write_index--;
                read_index--;
            }

            // update the index to the new position
            sync_vertices_[jj].index = new_index[jj - start_vertex];
        }
        // move current_vertex to the next position of ii
        curr_vertex = next_to_start;
    }
    

    for (int64_t curr_i = start_vertex; curr_i < end_vertex; curr_i++)
    {
        assert(test_index[curr_i - start_vertex] == sync_vertices_[curr_i].index && "Index is not correct!");
        for (int64_t i = 0; i < sync_vertices_[curr_i].degree; i++)
        {
            assert(test_edges[test_index[curr_i - start_vertex] + i] == edges_[sync_vertices_[curr_i].index + i] && "Edge is not correct!");
        }
    }

    free(test_index);
    free(test_edges);
    */
    
    free(copy_edges);
    copy_edges = nullptr;
    free(new_index);
    new_index = nullptr;

    recount_segment_total(start_vertex, end_vertex);
}

void PMAGraph::resize()
{
    // printf("[resize()] Called!\n");
    elem_capacity *= 2;
    int64_t gaps = elem_capacity - nedges;
    int64_t *new_indices = calculate_positions_V1(0, nnodes, gaps, nedges);
    int64_t *new_edges_ = (int64_t *)malloc(sizeof(int64_t) * elem_capacity);

    for (int64_t curr_vertex = nnodes - 1; curr_vertex >= 0; curr_vertex--)
    {
        for (size_t i = 0; i < sync_vertices_[curr_vertex].degree; i += 1)
        {
            new_edges_[new_indices[curr_vertex] + i] = edges_[sync_vertices_[curr_vertex].index + i];
            // edges_[new_indices[curr_vertex] + i] = edges_[vertices_[curr_vertex].index + i];
            // edges_[vertices_[curr_vertex].index + i] = -1;
        }
        sync_vertices_[curr_vertex].index = new_indices[curr_vertex];
    }
    for (int64_t edge = 0; edge < elem_capacity; edge++)
    {
        edges_[edge] = new_edges_[edge];
    }
    free(new_edges_);
    free(new_indices);
    // free(edges_);
    // edges_ = nullptr;
    // edges_ = new_edges_;
    recount_segment_total();
}

void PMAGraph::print_vertices()
{
    for (int64_t i = 0; i < nnodes; i++)
    {
        printf("dst: (%d)|%llu|%llu| \n    src=", i, sync_vertices_[i].index, sync_vertices_[i].degree);
        for (auto j = 0; j < sync_vertices_[i].degree; j++)
        {
            printf("%d, ", edges_[sync_vertices_[i].index + j]);
        }
        printf("\n");
    }
}

// sample

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PMAGraph::sample_neighbor(const torch::Tensor &vertices, int k)
{
    cudaStream_t stream = dev_stream[0];
    const size_t bs = vertices.size(0);
    thrust::device_vector<T> outputs;
    thrust::device_vector<T> output_counts(bs);

    sample_kernel(stream, vertices, k, outputs, output_counts);

    const auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<T> subset;
    PMAGraph::reindex_kernel(stream, vertices, outputs, subset);

    int tot = outputs.size();

    thrust::device_vector<T> input_prefix(output_counts);
    thrust::exclusive_scan(policy, input_prefix.begin(), input_prefix.end(),
                           input_prefix.begin());

    using it = thrust::counting_iterator<T>;

    torch::Tensor out_vertices = torch::empty(subset.size(), vertices.options());
    torch::Tensor row_idx = torch::empty(tot, vertices.options());
    torch::Tensor col_idx = torch::empty(tot, vertices.options());

    {
        thrust::device_vector<T> seq(output_counts.size());
        thrust::sequence(policy, seq.begin(), seq.end());

        thrust::for_each(
            policy, it(0), it(bs),
            [prefix = thrust::raw_pointer_cast(input_prefix.data()),
             count = thrust::raw_pointer_cast(output_counts.data()),
             in = thrust::raw_pointer_cast(seq.data()),
             out = row_idx.data_ptr<T>()] __device__(T i)
            {
                for (int j = 0; j < count[i]; j++)
                {
                    out[prefix[i] + j] = in[i];
                }
            });

        thrust::copy(policy, subset.begin(), subset.end(), out_vertices.data_ptr<T>());
        thrust::copy(policy, outputs.begin(), outputs.end(), col_idx.data_ptr<T>());
    }

    return std::make_tuple(out_vertices, row_idx, col_idx);
}

void PMAGraph::sample_kernel(const cudaStream_t stream,
                             const torch::Tensor &vertices, int k,
                             thrust::device_vector<T> &outputs,
                             thrust::device_vector<T> &output_counts) const
{
    T tot = 0;
    const auto policy = thrust::cuda::par.on(stream);
    const size_t bs = vertices.size(0);
    thrust::device_vector<T> output_ptr;
    thrust::device_vector<T> output_idx;

    {
        TRACE_SCOPE("alloc_1");
        output_counts.resize(bs);
        output_ptr.resize(bs);
    }
    // output_ptr is exclusive prefix sum of output_counts(neighbor counts <= k)
    {
        TRACE_SCOPE("prepare");
        thrust::counting_iterator<int> first(0);

        thrust::for_each(policy, first, first + bs,
                         [p = vertices.data_ptr<T>(),
                          o = thrust::raw_pointer_cast(output_counts.data()),
                          x = this->dev_sync_vertices_,
                          n = this->nnodes, cap = k] __device__(int i)
                         {
                             T key = p[i];
                             T num = key + 1 < n ? x[key].degree : 0;
                             if (cap >= 0 && num > cap)
                             {
                                 num = cap;
                             }
                             o[i] = num;
                         });
        TRACE_SCOPE("exclusive_scan");
        thrust::exclusive_scan(policy, output_counts.begin(),
                               output_counts.end(), output_ptr.begin());
        TRACE_SCOPE("reduce");
        tot = thrust::reduce(policy, output_counts.begin(),
                             output_counts.end());
    }
    {
        TRACE_SCOPE("alloc_2");
        outputs.resize(tot);
        output_idx.resize(tot);
    }
    // outputs[outptr[i], outptr[i + 1]) are unique neighbors of inputs[i]
    {
        TRACE_SCOPE("sample");
        new_sample(stream, k, vertices.data_ptr<T>(), bs,
                   thrust::raw_pointer_cast(output_ptr.data()),
                   thrust::raw_pointer_cast(output_counts.data()),
                   thrust::raw_pointer_cast(outputs.data()),
                   thrust::raw_pointer_cast(output_idx.data()));
    }
}

void PMAGraph::new_sample(const cudaStream_t stream, int k, T *input_begin,
                          int64_t input_size, T *output_ptr_begin, T *output_count_begin,
                          T *output_begin, T *output_idx) const
{
    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((input_size + TILE_SIZE - 1) / TILE_SIZE);
    CSRRowWiseSampleKernel<T, vertex_sync_element_t, BLOCK_WARPS, TILE_SIZE>
        <<<grid, block, 0, stream>>>(
            0, k, input_size, input_begin,
            this->dev_sync_vertices_,
            this->dev_edges_,
            output_ptr_begin, output_count_begin,
            output_begin, output_idx);
}

// return counts for each vertex
void PMAGraph::reindex_kernel(const cudaStream_t stream,
                              const torch::Tensor &vertices,
                              thrust::device_vector<T> &outputs,
                              thrust::device_vector<T> &subset)
{
    const auto policy = thrust::cuda::par.on(stream);
    HostOrderedHashTable<T> *table;
    // reindex
    {
        {
            TRACE_SCOPE("reindex 0");
            const size_t bs = vertices.size(0);
            subset.resize(bs + outputs.size());

            thrust::counting_iterator<int> first(0);
            thrust::for_each(policy, first, first + subset.size(),
                             [p = vertices.data_ptr<T>(),
                              o = thrust::raw_pointer_cast(outputs.data()),
                              s = thrust::raw_pointer_cast(subset.data()),
                              bs_ = bs] __device__(int i)
                             {
                                 if (i < bs_)
                                 {
                                     s[i] = p[i];
                                 }
                                 else
                                 {
                                     s[i] = o[i - bs_];
                                 }
                             });

            thrust::device_vector<T> unique_items;

            unique_items.clear();
            table =
                FillWithDuplicates(thrust::raw_pointer_cast(subset.data()),
                                   subset.size(), stream, unique_items);
            // subset.resize(unique_items.size());
            // thrust::copy(policy, unique_items.begin(), unique_items.end(),
            //              subset.begin());
            subset = unique_items;
        }
        {
            TRACE_SCOPE("permute");
            DeviceOrderedHashTable<T> device_table = table->DeviceHandle();
            thrust::for_each(
                policy, outputs.begin(), outputs.end(),
                [device_table] __device__(T & id) mutable
                {
                    using Iterator =
                        typename DeviceOrderedHashTable<T>::Iterator;
                    Iterator iter = device_table.Search(id);
                    id = static_cast<T>((*iter).local);
                });
        }
        delete table;
    }
}


torch::Tensor PMAGraph::next_hops(const torch::Tensor &vertices)
{
    const auto policy = thrust::cuda::par.on(dev_stream[0]);
    thrust::device_vector<int> col_count(vertices.size(0) + 1, 0);
    thrust::counting_iterator<int> first(0);

    col_count[0] = vertices.size(0);
    thrust::for_each(policy, first, first + vertices.size(0),
                     [col_c = thrust::raw_pointer_cast(col_count.data()),
                      k = vertices.data_ptr<T>(),
                      x = this->dev_sync_vertices_] __device__(int i)
                     {
                         T key = k[i];
                         col_c[i + 1] = x[key].degree;
                     });

    thrust::inclusive_scan(policy, col_count.begin(), col_count.end(), col_count.begin());

    int size_num = col_count.back();
    torch::Tensor out_vertices = torch::empty(size_num, vertices.options());
    thrust::for_each(policy, first, first + vertices.size(0),
                     [col_c = thrust::raw_pointer_cast(col_count.data()),
                      k = vertices.data_ptr<T>(),
                      x = this->dev_sync_vertices_,
                      e = this->dev_edges_,
                      out = out_vertices.data_ptr<T>()] __device__(int i)
                     {
                         T key = k[i];
                         out[i] = key;
                         for (int j = 0; j < x[key].degree; j++)
                         {
                             out[col_c[i] + j] = e[x[key].index + j];
                         }
                     });

    return out_vertices;
}

std::tuple<torch::Tensor, torch::Tensor>
PMAGraph::affected_edges(const torch::Tensor &vertices) {
    const auto policy = thrust::cuda::par.on(dev_stream[0]);
    thrust::device_vector<int> col_count(vertices.size(0) + 1, 0);
    thrust::counting_iterator<int> first(0);

    thrust::for_each(
        policy, first, first + vertices.size(0), [
            col_cnt = thrust::raw_pointer_cast(col_count.data()),
            vertices = vertices.data_ptr<T>(),
            dev_sync_vertices = this->dev_sync_vertices_
        ] __device__(int i) {
            T key = vertices[i];
            col_cnt[i + 1] = dev_sync_vertices[key].degree;
        });

    thrust::inclusive_scan(policy, col_count.begin(), col_count.end(), col_count.begin());

    int size_num = col_count.back();
    torch::Tensor src = torch::empty(size_num, vertices.options());
    torch::Tensor dst = torch::empty(size_num, vertices.options());

    thrust::for_each(
        policy, first, first + vertices.size(0), [
            col_cnt = thrust::raw_pointer_cast(col_count.data()),
            vertices = vertices.data_ptr<T>(),
            dev_sync_vertices = this->dev_sync_vertices_,
            dev_edges = this->dev_edges_,
            src = src.data_ptr<T>(),
            dst = dst.data_ptr<T>(),
            num_vertices = vertices.size(0)
        ] __device__(int i) {
            int warp_id = threadIdx.x / WARP_SIZE; // 每个 warp 处理一个顶点
            int lane_id = threadIdx.x % WARP_SIZE; // warp 内的线程 ID
            int global_id = blockIdx.x * blockDim.x + threadIdx.x;
            int NUM_WARPS = blockDim.x / WARP_SIZE;

            for(int vtx_idx = warp_id; vtx_idx < num_vertices; vtx_idx += NUM_WARPS) {
                // 读取当前 warp 处理的顶点
                T key = __ldg(&vertices[vtx_idx]);
                // 获取顶点的度数和邻接表索引
                int degree = __ldg(&dev_sync_vertices[key].degree);
                int edge_start = __ldg(&col_cnt[vtx_idx]);
                // 使用 warp 处理所有邻居
                for (int j = lane_id; j < degree; j += WARP_SIZE) {
                    int index = edge_start + j;
                    src[index] = key;
                    dst[index] = __ldg(&dev_edges[dev_sync_vertices[key].index + j]);
                }
            }
        });

    // thrust::for_each(policy, first, first + vertices.size(0),
    //                 [col_c = thrust::raw_pointer_cast(col_count.data()),
    //                 k = vertices.data_ptr<T>(),
    //                 x = this->dev_sync_vertices_,
    //                 e = this->dev_edges_,
    //                 s = src.data_ptr<T>(),
    //                 d = dst.data_ptr<T>()] __device__(int i)
    //                 {
    //                     T key = k[i];
    //                     for (int j = 0; j < x[key].degree; j++)
    //                     {
    //                         int index = col_c[i] + j;
    //                         s[index] = key;
    //                         d[index] = e[x[key].index + j];
    //                     }
    //                 });

    return std::make_tuple(src, dst);
}

// src nodes -> dst nodes type (normal, insert, delete)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PMAGraph::next_layer(const torch::Tensor &insert_edges, 
                     const torch::Tensor &delete_edges, 
                     const torch::Tensor &vertices)
{
    // async
    const auto policy = thrust::cuda::par.on(dev_stream[0]);
    thrust::device_vector<int> col_count(vertices.size(0) + 1, 0);
    thrust::counting_iterator<int> first(0);

    thrust::for_each(policy, first, first + vertices.size(0),
                    [col_c = thrust::raw_pointer_cast(col_count.data()),
                     k = vertices.data_ptr<T>(),
                     x = this->dev_sync_vertices_] __device__(int i)
                    {
                        T key = k[i];
                        col_c[i + 1] = x[key].degree;
                    });

    thrust::inclusive_scan(policy, col_count.begin(), col_count.end(), col_count.begin());

    int size_num = col_count.back() + insert_edges.size(1) + delete_edges.size(1);
    torch::Tensor src = torch::empty(size_num, vertices.options());
    torch::Tensor dst = torch::empty(size_num, vertices.options());
    torch::Tensor type = torch::zeros(size_num, vertices.options().dtype(torch::kInt8));

    thrust::for_each(policy, first, first + vertices.size(0),
                    [col_c = thrust::raw_pointer_cast(col_count.data()),
                     k = vertices.data_ptr<T>(),
                     x = this->dev_sync_vertices_,
                     e = this->dev_edges_,
                     s = src.data_ptr<T>(),
                     d = dst.data_ptr<T>()] __device__(int i)
                    {
                        T key = k[i];
                        for (int j = 0; j < x[key].degree; j++)
                        {
                            int index = col_c[i] + j;
                            s[index] = key;
                            d[index] = e[x[key].index + j];
                        }
                    });

    size_num = col_count.back();
    thrust::copy_n(policy, insert_edges.data_ptr<T>(), 
        insert_edges.size(1), 
        src.data_ptr<T>() + size_num);
    thrust::copy_n(policy, insert_edges.data_ptr<T>() + insert_edges.size(1), 
        insert_edges.size(1), 
        dst.data_ptr<T>() + size_num);
    thrust::fill_n(policy, type.data_ptr<int8_t>() + size_num, 
        insert_edges.size(1), 1);

    size_num += insert_edges.size(1);
    thrust::copy_n(policy, delete_edges.data_ptr<T>(), 
        delete_edges.size(1), 
        src.data_ptr<T>() + size_num);
    thrust::copy_n(policy, delete_edges.data_ptr<T>() + delete_edges.size(1), 
        delete_edges.size(1), 
        dst.data_ptr<T>() + size_num);
    thrust::fill_n(policy, type.data_ptr<int8_t>() + size_num, 
        delete_edges.size(1), -1);

    return std::make_tuple(src, dst, type);
}

// indices, dst
std::tuple<torch::Tensor, torch::Tensor>
PMAGraph::next_layer_indices(const torch::Tensor &vertices, const torch::Tensor &ignore)
{
    // async
    const auto policy = thrust::cuda::par.on(dev_stream[0]);
    thrust::device_vector<int> col_count(vertices.size(0) + 1, 0);
    thrust::counting_iterator<int> first(0);

    thrust::for_each(policy, first, first + vertices.size(0),
                    [col_c = thrust::raw_pointer_cast(col_count.data()),
                     ig = ignore.size(0) >= vertices.size(0),
                     l = ignore.data_ptr<bool>(),
                     k = vertices.data_ptr<T>(),
                     x = this->dev_sync_vertices_] __device__(int i)
                    {
                        T key = k[i];
                        if (ig && l[i])
                        {
                            col_c[i + 1] = 0;
                        }
                        else
                        {
                            col_c[i + 1] = x[key].degree;
                        }
                    });

    thrust::inclusive_scan(policy, col_count.begin(), col_count.end(), col_count.begin());

    torch::Tensor indices = torch::empty(col_count.size(), vertices.options());
    thrust::copy(policy, col_count.begin(), col_count.end(), indices.data_ptr<T>());

    int size_num = col_count.back();
    torch::Tensor dst = torch::empty(size_num, vertices.options());
    thrust::for_each(policy, first, first + vertices.size(0),
                    [col_c = thrust::raw_pointer_cast(col_count.data()),
                     k = vertices.data_ptr<T>(),
                     ig = ignore.size(0) >= vertices.size(0),
                     l = ignore.data_ptr<bool>(),
                     x = this->dev_sync_vertices_,
                     e = this->dev_edges_,
                     d = dst.data_ptr<T>()] __device__(int i)
                    {
                        T key = k[i];
                        int degree = 0;
                        if (!(ig && l[i])) {
                            degree = x[key].degree;
                        }
                        for (int j = 0; j < degree; j++)
                        {
                            int index = col_c[i] + j;
                            d[index] = e[x[key].index + j];
                        }
                    });

    return std::make_tuple(indices, dst);
}

bool PMAGraph::transpose(torch::Tensor &t_indexes, torch::Tensor &t_nodes)
{

    if (t_indexes.size(0) <= this->nnodes || t_nodes.size(0) < this->nedges || t_indexes.get_device() < 0 || t_nodes.get_device() < 0)
    {
        return false;
    }

    const auto policy = thrust::cuda::par.on(dev_stream[0]);
    thrust::device_vector<int> col_count(this->nnodes, 0);
    thrust::counting_iterator<int> first(0);

    thrust::for_each(policy, first, first + this->nnodes,
                     [col_c = thrust::raw_pointer_cast(col_count.data()),
                      x = this->dev_sync_vertices_,
                      v = this->dev_edges_] __device__(int i)
                     {
                         for (int j = 0; j < x[i].degree; j++)
                         {
                             atomicAdd(&col_c[v[x[i].index + j]], 1);
                         }
                     });

    thrust::device_vector<int> start_pos(this->nnodes + 1, 0);

    for (int i = 1; i < start_pos.size(); ++i)
    {
        start_pos[i] = start_pos[i - 1] + col_count[i - 1];
    }

    thrust::copy(policy, start_pos.begin(), start_pos.end(), t_indexes.data_ptr<int64_t>());

    thrust::for_each(policy, first, first + this->nnodes,
                     [x = this->dev_sync_vertices_,
                      v = this->dev_edges_,
                      s = thrust::raw_pointer_cast(start_pos.data()),
                      t = t_nodes.data_ptr<int64_t>()] __device__(int i)
                     {
                         for (int j = 0; j < x[i].degree; j++)
                         {
                             int64_t dest = v[x[i].index + j];
                             int pos = atomicAdd(&s[dest], 1);
                             t[pos] = i;
                         }
                     });

    return true;
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<PMAGraph>(m, "PMAGraph")
        .def(py::init<int64_t, int64_t, const torch::Tensor &, const torch::Tensor &, int, int64_t>())
        .def("sample_neighbor", &PMAGraph::sample_neighbor)
        .def("insert_edge", &PMAGraph::insert_edge)
        .def("delete_edge", &PMAGraph::delete_edge)
        .def("num_edges", &PMAGraph::num_edges)
        .def("num_nodes", &PMAGraph::num_nodes)
        .def("has_edge", &PMAGraph::has_edge)
        .def("transpose", &PMAGraph::transpose)
        .def("next_hops", &PMAGraph::next_hops)
        .def("affected_edges", &PMAGraph::affected_edges)
        .def("next_layer", &PMAGraph::next_layer)
        .def("next_layer_indices", &PMAGraph::next_layer_indices)
        .def("print_vertices", &PMAGraph::print_vertices);
}
