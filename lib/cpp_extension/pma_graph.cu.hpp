#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/system/cuda/execution_policy.h>

typedef struct vertex_sync_element
{
    int64_t index;
    int64_t degree;
} vertex_sync_element_t;

typedef struct vertex_element
{
    // int64_t third_start;
    // int64_t third_degree;
    // bool delta;
    // bool cache;
    // bool deletion;
    int64_t cache_index;
    int64_t host_index;
    uint8_t hotness[4];
    bool data_valid;
    // int64_t virtual_start;
    // int64_t virtual_degree;
    // int64_t secondary_start;
    // int64_t secondary_degree;
} vertex_element_t;

class PMAGraph 
{
    using T = int64_t;
    
public:
    // PMAGraph(int64_t nnodes, int64_t nedges) : row_start_vec(nnodes + 1), edge_dst_vec(nedges)
    // {
    //     row_start = &row_start_vec[0];
    //     edge_dst = &edge_dst_vec[0];
    // }

    PMAGraph(int64_t _nnodes, int64_t _nedges,
             const torch::Tensor &row_index, const torch::Tensor &col_idx,
             int device_id, int64_t indicated_nedges) : cuda_device(device_id)
    {
        cudaSetDevice(cuda_device);

        for (int i = 0; i < NUM_STREAM; i++) {
            cudaError_t err = cudaStreamCreate(&dev_stream[i]);
            if (err != cudaSuccess) {
                std::cerr << "Error creating stream: " << cudaGetErrorString(err) << "\n";
            }
        }
        
        this->Bind(_nnodes, _nedges, row_index.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), indicated_nedges);

        cudaHostGetDevicePointer(&dev_sync_vertices_, sync_vertices_, 0);
        cudaHostGetDevicePointer(&dev_edges_, edges_, 0);

    }

    ~PMAGraph()
    {
        if (segment_edges_actual != nullptr)
            free(segment_edges_actual);
        if (segment_edges_total != nullptr)
            free(segment_edges_total);
        if (edges_ != nullptr)
            cudaFreeHost(edges_);
        // if (vertices_ != nullptr)
        //     cudaFreeHost(vertices_);
        if (sync_vertices_ != nullptr)
            cudaFreeHost(sync_vertices_);

        for (int i = 0; i < NUM_STREAM; i++) {
            cudaError_t err = cudaStreamDestroy(dev_stream[i]);
            if (err != cudaSuccess) {
                std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << "\n";
            }
        }
    }

    int64_t begin_edge(int64_t node) const
    {
        return sync_vertices_[node].index;
    }

    int64_t end_edge(int64_t node) const
    {
        return sync_vertices_[node + 1].index;
    }

    int64_t edge_degree(int64_t node) const
    {
        return sync_vertices_[node].degree;
    }

    int64_t edge_dest(int64_t edge) const
    {
        return edges_[edge];
    }

    void insert_edge(int64_t src, int64_t dst, int64_t value = 0);

    void delete_edge(int64_t src, int64_t dst, int64_t weight = 0);

    void print_vertices();

    int64_t num_edges() { return nedges; }

    int64_t num_nodes() { return nnodes; }

    bool has_edge(int64_t src, int64_t dst) {
        int64_t start = begin_edge(src);
        for (int64_t i = 0; i < edge_degree(src); i++) {
            if (edge_dest(start + i) == dst) {
                return true;
            }
        }
        return false;
    }

private:

    constexpr static int NUM_STREAM = 1;

    int cuda_device;
    cudaStream_t dev_stream[NUM_STREAM];
    
    // int64_t *row_start;
    // int64_t *edge_dst;

    // std::vector<int64_t> row_start_vec; // the vectors are not always in use (see Bind)
    // std::vector<int64_t> edge_dst_vec;
    // std::vector<int64_t> edge_weights_vec;
    // int32_t *subgraph_activenode;
    // int32_t *subgraph_rowstart;
    // int32_t *subgraph_edgedst;
    // int32_t *subgraph_edgeweight;
    int64_t nnodes = 0;
    int64_t nedges = 0;
    // parameters of pma code by yongze
    int64_t elem_capacity = 0;
    int64_t elem_capacity_max = 0;
    int32_t segment_size = 0;
    int64_t segment_count = 0;
    int32_t tree_height = 0;
    double delta_up;  // Delta for upper density threshold
    double delta_low; // Delta for lower density threshold
    
    // vertex_element_t *vertices_;
    vertex_sync_element_t *sync_vertices_; // index and degree (begin and end)
    int64_t *edges_ = nullptr;                  // adj list
    int64_t *weights_ = nullptr;

    vertex_sync_element_t *dev_sync_vertices_; 
    int64_t *dev_edges_;             
    int64_t *dev_weights_;
    
    int64_t *segment_edges_actual = nullptr;
    int64_t *segment_edges_total = nullptr;
    bool *delta = nullptr;
    static constexpr double up_h = 0.75; // root
    static constexpr double up_0 = 1.00; // leaves
    // Lower density thresholds
    static constexpr double low_h = 0.50; // root
    static constexpr double low_0 = 0.25; // leaves
    int64_t river = 0;
    int64_t river_low = 0;
    int64_t river_global = 0;
    int8_t max_sparseness = 0.5 / low_0;
    int8_t largest_empty_segment = 1.0 * max_sparseness;

    void compute_capacity(int64_t indicated_size = 0);

    void spread_weighted_V1(int32_t start_vertex, int32_t end_vertex);

    void recount_segment_total();

    void recount_segment_total(int64_t start_vertex, int64_t end_vertex);

    int64_t *calculate_positions_V1(int64_t start_vertex, int64_t end_vertex, int64_t gaps, int64_t total_degree);

    void rebalance_wrapper(int64_t src);

    void rebalance_weighted(int64_t start_vertex, int64_t end_vertex, int64_t pma_idx);

    void resize();

    int64_t avg_degree() const
    {
        return nedges / nnodes;
    }

    int64_t get_segment_id(int64_t vertex_id)
    {
        return (vertex_id / segment_size) + segment_count;
    }

    void update_segment_edge_total(int64_t vertex_id, int count)
    {
        int64_t j = get_segment_id(vertex_id);
        while (j > 0)
        {
            segment_edges_total[j] += count;
            j /= 2;
        }
    }


    void Bind(int64_t nnodes, int64_t nedges,
              int64_t *row_start, int64_t *edge_dst, 
              int64_t indicated_size = 0); 


    // sample
public:

    inline void sample_kernel(const cudaStream_t stream, const torch::Tensor &vertices,
                  int k, // thrust::device_vector<T> &inputs,
                  thrust::device_vector<T> &outputs,
                  thrust::device_vector<T> &output_counts) const;


    inline void new_sample(const cudaStream_t stream, int k, T *input_begin,
                    int64_t input_size, T *output_ptr_begin, T *output_count_begin,
                    T *output_begin, T *output_idx) const;

    inline static void reindex_kernel(const cudaStream_t stream,
                const torch::Tensor &vertices, 
                thrust::device_vector<T> &outputs,
                thrust::device_vector<T> &subset);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    sample_neighbor(const torch::Tensor &vertices, int k);

    bool transpose(torch::Tensor &indexes, torch::Tensor &nodes);

    torch::Tensor next_hops(const torch::Tensor &vertices);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    next_layer(const torch::Tensor &insert_edges, 
               const torch::Tensor &delete_edges, 
               const torch::Tensor &vertices);

    std::tuple<torch::Tensor, torch::Tensor>
    affected_edges(const torch::Tensor &vertices);
    
    std::tuple<torch::Tensor, torch::Tensor>
    next_layer_indices(const torch::Tensor &vertices, const torch::Tensor &ignore);

};
