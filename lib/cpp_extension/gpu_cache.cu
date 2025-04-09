#include "pma_graph.cu.hpp"


// template <typename key_t>
class GPUDataCache
{
    constexpr static int WARP_SIZE = 32;
    constexpr static int64_t INVALID = -1;
    constexpr static int NUM_STREAM = 3;

private:
    size_t num_host_items;
    float *host_cache;
    
    size_t num_items;
    float *device_cache;

    // thrust::device_vector<int> cache_nodes;
    // thrust::device_vector<int> hotness;

    vertex_element_t *device_cache_index;
    void* host_ptr;
    size_t num_nodes;
    size_t num_feats;
    int cuda_device;
    cudaStream_t stream[NUM_STREAM]; // 0 for query and 1 for replace

public:

    inline __device__ static void data_copy(float* out_ptr, float* in_ptr, int num_feats) {
        // for (int64_t i = 0; i < num_feats; i++) {
        //     out_ptr[i] = in_ptr[i];
        // }

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int warp_idx = idx / 32;                           // warp idx
        int lane_idx = idx % 32;                           // idx in warp
    
        if (warp_idx < num_feats / 32) {
            // every warp access a memory lane
            out_ptr[warp_idx * 32 + lane_idx] = in_ptr[warp_idx * 32 + lane_idx];
        }
    }

    void init(const torch::Tensor& host_key, const torch::Tensor& device_key) {
        const auto policy0 = thrust::cuda::par.on(stream[0]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + num_nodes, 
            [index = this->device_cache_index] __device__ (int64_t i) {
                index[i].cache_index = INVALID;
                index[i].host_index = INVALID;
                index[i].data_valid = true;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    index[i].hotness[j] = 0;
            });
        
        thrust::for_each(policy0, first, first + num_host_items, 
            [index = this->device_cache_index,
             keys_data = host_key.data_ptr<int64_t>()
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                index[key].host_index = i;
            });

        thrust::for_each(policy0, first, first + num_items,
            [index = this->device_cache_index,
             keys_data = device_key.data_ptr<int64_t>()
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                index[key].cache_index = i;
            });
    }

    GPUDataCache(size_t num_nodes_, 
        const torch::Tensor& host_key, const torch::Tensor& host_embedding, 
        const torch::Tensor& device_key, const torch::Tensor& device_embedding, 
        int device_id, bool in_gpu = true)
        : num_items(device_key.size(0)), num_nodes(num_nodes_), 
          num_feats(host_embedding.size(1)),
          num_host_items(host_key.size(0)), cuda_device(device_id)
    {
        assert(device_embedding.get_device() >= 0 && "device_embedding should be in device!");
        
        cudaSetDevice(cuda_device);
        device_cache = device_embedding.data_ptr<float>();
        if (in_gpu) {
            cudaMalloc(&device_cache_index, num_nodes * sizeof(vertex_element_t));
            host_ptr = nullptr;
        } else {
            cudaMallocHost(&host_ptr, num_nodes * sizeof(vertex_element_t));
            cudaHostGetDevicePointer(&device_cache_index, host_ptr, 0);
        }

        auto ptr_host = host_embedding.data_ptr<float>();
        cudaHostGetDevicePointer(&host_cache, ptr_host, 0);
        for (int64_t i = 0; i < NUM_STREAM; i++) {
            cudaError_t err = cudaStreamCreate(&stream[i]);
            if (err != cudaSuccess) {
                std::cerr << "Error creating stream: " << cudaGetErrorString(err) << "\n";
            }
        }
    
        init(host_key, device_key);
    }

    ~GPUDataCache()
    {
        // cudaFree(device_cache);
        if (host_ptr != nullptr) {
            cudaFreeHost(host_ptr);
        } else {
            cudaFree(device_cache_index);
        }
        // cudaFreeHost(host_cache);
        for (int64_t i = 0; i < NUM_STREAM; i++) {
            cudaError_t err = cudaStreamDestroy(stream[i]);
            if (err != cudaSuccess) {
                std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << "\n";
            }
        }            
    }

    torch::Tensor check_state(const torch::Tensor &keys, bool ignore_valid = true) {
        torch::Tensor out = torch::zeros_like(keys, keys.options().dtype(torch::kInt8));
        const auto policy0 = thrust::cuda::par.on(stream[0]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + keys.size(0), 
            [ignore = ignore_valid,
             num_items_ = this->num_items,
             num_hosts_ = this->num_host_items,
             cache_index = this->device_cache_index,
             out_base = out.data_ptr<int8_t>(),
             keys_data = keys.data_ptr<int64_t>()
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                if (!ignore && !cache_index[key].data_valid) {
                    out_base[i] = -1;
                    return;
                }
                int64_t device_index = cache_index[key].cache_index;
                int64_t host_index = cache_index[key].host_index;
                if (device_index > INVALID && device_index < num_items_) {
                    out_base[i] = 1;
                } else if (host_index > INVALID && host_index < num_hosts_) {
                    out_base[i] = 0;
                } else {
                    out_base[i] = -1;
                }
            });

        return out;
    }

    bool query(const torch::Tensor &keys, const torch::Tensor &counts, 
               torch::Tensor& out_values, torch::Tensor& missing, 
               bool ignore_valid = true) {
        if (keys.size(0) != out_values.size(0) || out_values.size(1) != num_feats || out_values.get_device() < 0) {
            return false;
        }

        const auto policy0 = thrust::cuda::par.on(stream[0]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + keys.size(0), 
            [ignore_ = ignore_valid,
             num_feats_ = this->num_feats,
             num_items_ = this->num_items,
             num_hosts_ = this->num_host_items,
             cache_index = this->device_cache_index,
             out_base = out_values.data_ptr<float>(),
             cache_base = this->device_cache,
             host_base = this->host_cache,
             keys_data = keys.data_ptr<int64_t>(),
             counts_data = counts.data_ptr<int64_t>(),
             missing_data = missing.data_ptr<bool>()
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                if (!ignore_ && !cache_index[key].data_valid) {
                    missing_data[i] = true;
                } else {
                int64_t device_index = cache_index[key].cache_index;
                int64_t host_index = cache_index[key].host_index;
                float* out_ptr = out_base + i * num_feats_;
                float* in_ptr = nullptr;

                if (device_index > INVALID && device_index < num_items_) {
                    in_ptr = cache_base + device_index * num_feats_;
                    data_copy(out_ptr, in_ptr, num_feats_);
                    missing_data[i] = false;
                } else if (host_index > INVALID && host_index < num_hosts_) {
                    in_ptr = host_base + host_index * num_feats_;
                    data_copy(out_ptr, in_ptr, num_feats_);
                    missing_data[i] = false;
                } else {
                    missing_data[i] = true;
                }}
                cache_index[key].hotness[3] += counts_data[i];
            });
        return true;
    }

    // three stream: one for query
    //                  in cache: copy 
    //                  not in cache: zore copy from host 
    //             one for computing hotness and replace

    bool invalidate(const torch::Tensor &keys) {
        const auto policy0 = thrust::cuda::par.on(stream[0]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + keys.size(0), 
            [cache_index = this->device_cache_index,
             keys_data = keys.data_ptr<int64_t>()
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                cache_index[key].data_valid = false;
            });
        return true;
    }

    bool update_hotness(torch::Tensor &hotness) {
        const auto policy0 = thrust::cuda::par.on(stream[1]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + num_nodes, 
            [cache_index = this->device_cache_index,
             cache_hotness = hotness.data_ptr<int64_t>()
             ] __device__ (int64_t i) {
                cache_hotness[i] -= cache_index[i].hotness[0];
                cache_hotness[i] += cache_index[i].hotness[3];
                // int32_t* hot_ptr = (int32_t*)cache_index[i].hotness;
                // *hot_ptr <<= 8;
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    cache_index[i].hotness[j] = cache_index[i].hotness[j + 1];
                }
                cache_index[i].hotness[3] = 0;
            });
    }

    bool replace_from_host(const torch::Tensor &device_remove, const torch::Tensor &device_insert) {

        const auto policy0 = thrust::cuda::par.on(stream[1]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + device_insert.numel(), 
            [ls_remove = device_remove.data_ptr<int64_t>(),
             ls_insert = device_insert.data_ptr<int64_t>(),
             num_feats_ = this->num_feats,
             cache_index = this->device_cache_index,
             cache_base = this->device_cache,
             host_base = this->host_cache
            ] __device__ (int64_t i) {
                int64_t remove_key = ls_remove[i];
                int64_t insert_key = ls_insert[i];
                int64_t cache_pos = cache_index[remove_key].cache_index;
                int64_t host_pos = cache_index[insert_key].host_index;

                cache_index[remove_key].cache_index = INVALID;
                cache_index[insert_key].cache_index = cache_pos;
                
                auto out_ptr = cache_base + cache_pos * num_feats_;
                auto in_ptr = host_base + host_pos * num_feats_;
                data_copy(out_ptr, in_ptr, num_feats_);
            });

        return true;
    }


    inline bool replace_dev(
        const thrust::device_vector<int64_t*> &dev_keys, 
        const thrust::device_vector<int64_t> &dev_num_keys,
        const thrust::device_vector<float*> &dev_values,
        const torch::Tensor &device_remove, const torch::Tensor &device_insert, const torch::Tensor &device_pos, const torch::Tensor &host_values) {

        const auto policy0 = thrust::cuda::par.on(stream[1]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + device_insert.numel(), 
            [ls_remove = device_remove.data_ptr<int64_t>(),
             ls_insert = device_insert.data_ptr<int64_t>(),
             ls_pos = device_pos.data_ptr<int64_t>(),
             ls_key = thrust::raw_pointer_cast(dev_keys.data()),
             key_size = thrust::raw_pointer_cast(dev_num_keys.data()),
             value_base = thrust::raw_pointer_cast(dev_values.data()),
             num_keys_ = dev_keys.size(),
             num_feats_ = this->num_feats,
             cache_index = this->device_cache_index,
             cache_base = this->device_cache,
             host_values_base = host_values.data_ptr<float>(),
             host_size = host_values.size(0)
            ] __device__ (int64_t i) {
                int64_t remove_key = ls_remove[i];
                int64_t insert_key = ls_insert[i];
                int64_t data_pos = ls_pos[i];
                int64_t cache_pos = cache_index[remove_key].cache_index;

                cache_index[remove_key].cache_index = INVALID;
                cache_index[insert_key].cache_index = cache_pos;                
                
                auto out_ptr = cache_base + cache_pos * num_feats_;
                float* in_ptr = nullptr;

                for (int j = 0; j < num_keys_; j++) {
                    if (data_pos < key_size[j] && ls_key[j][data_pos] == insert_key) {
                        in_ptr = value_base[j] + data_pos * num_feats_;
                        break;
                    }
                }
                if (in_ptr == nullptr) {
                    if (data_pos < host_size) {
                        in_ptr = host_values_base + data_pos * num_feats_;
                    } else {
                        printf("Error: replace_dev %lld %lld %lld %lld\n", 
                        i, data_pos, insert_key, remove_key);
                    }
                }

                data_copy(out_ptr, in_ptr, num_feats_);
            });

        return true;
    }

    bool replace(const std::vector<torch::Tensor> &ls_keys, 
        const std::vector<torch::Tensor> &ls_values, 
        const torch::Tensor &host_remove, const torch::Tensor &host_insert, const torch::Tensor &host_pos, 
        const torch::Tensor &device_remove, const torch::Tensor &device_insert, const torch::Tensor &device_pos, const torch::Tensor &host_values) {

        const auto policy0 = thrust::cuda::par.on(stream[1]);
        thrust::counting_iterator<int64_t> first(0);

        size_t num_keys = ls_keys.size();
        thrust::device_vector<int64_t*> dev_keys(num_keys);
        thrust::device_vector<int64_t> dev_num_keys(num_keys);
        thrust::device_vector<float*> dev_values(num_keys);
        for (int i = 0; i < num_keys; i++) {
            dev_keys[i] = ls_keys[i].data_ptr<int64_t>();
            dev_num_keys[i] = ls_keys[i].numel();
            float* value_ptr;
            if (ls_values[i].get_device() >= 0) {
                value_ptr = ls_values[i].data_ptr<float>();
            } else {
                cudaHostGetDevicePointer(&value_ptr, ls_values[i].data_ptr<float>(), 0);
            }
            dev_values[i] = value_ptr;
        }

        thrust::for_each(policy0, first, first + host_insert.numel(), 
            [ls_remove = host_remove.data_ptr<int64_t>(),
             ls_insert = host_insert.data_ptr<int64_t>(),
             ls_pos = host_pos.data_ptr<int64_t>(),
             ls_key = thrust::raw_pointer_cast(dev_keys.data()),
             key_size = thrust::raw_pointer_cast(dev_num_keys.data()),
             value_base = thrust::raw_pointer_cast(dev_values.data()),
             num_keys_ = num_keys,
             num_feats_ = this->num_feats,
             cache_index = this->device_cache_index,
             cache_base = this->device_cache,
             host_base = this->host_cache,
             num_items_ = this->num_items
             ] __device__ (int64_t i) {
                int64_t remove_key = ls_remove[i];
                int64_t insert_key = ls_insert[i];
                int64_t data_pos = ls_pos[i];
                int64_t cache_pos = cache_index[remove_key].host_index;

                cache_index[remove_key].host_index = INVALID;
                cache_index[insert_key].host_index = cache_pos;
                
                auto out_ptr = host_base + cache_pos * num_feats_;
                float* in_ptr = nullptr;
                
                for (int j = 0; j < num_keys_; j++) {
                    if (data_pos < key_size[j] && ls_key[j][data_pos] == insert_key) {
                        in_ptr = value_base[j] + data_pos * num_feats_;
                        break;
                    }
                }

                if (in_ptr == nullptr) {
                    data_pos = cache_index[insert_key].cache_index;
                    if (data_pos > INVALID && data_pos < num_items_) {
                        in_ptr = cache_base + data_pos * num_feats_;
                    } else {
                        printf("Error: replace %lld %lld %lld\n", 
                            data_pos, insert_key, remove_key);
                    }
                } 
                data_copy(out_ptr, in_ptr, num_feats_);
            });

        return replace_dev(dev_keys, dev_num_keys, dev_values, 
                    device_remove, device_insert, device_pos, host_values);
    }

    void sync_replace() {
        cudaStreamSynchronize(stream[1]);
    }

    bool update(const torch::Tensor &keys, const torch::Tensor &values, const torch::Tensor &ignore) {
        // const auto policy0 = thrust::cuda::par_nosync.on(stream[2]);
        const auto policy0 = thrust::cuda::par.on(stream[2]);
        thrust::counting_iterator<int64_t> first(0);

        thrust::for_each(policy0, first, first + keys.size(0), 
            [ig = ignore.size(0) >= keys.size(0),
             l = ignore.data_ptr<bool>(),
             keys_data = keys.data_ptr<int64_t>(),
             values_data = values.data_ptr<float>(),
             num_feats_ = this->num_feats,
             num_items_ = this->num_items,
             num_hosts_ = this->num_host_items,
             cache_index = this->device_cache_index,
             cache_base = this->device_cache,
             host_base = this->host_cache
             ] __device__ (int64_t i) {
                int64_t key = keys_data[i];
                int64_t device_index = cache_index[key].cache_index;
                int64_t host_index = cache_index[key].host_index;
                float* in_ptr = values_data + i * num_feats_;
                float* out_ptr;

                cache_index[key].data_valid = true;
                if (ig && l[i]) {
                    return;
                }
                if (device_index > INVALID && device_index < num_items_) {
                    out_ptr = cache_base + device_index * num_feats_;
                    data_copy(out_ptr, in_ptr, num_feats_);
                }
                if (host_index > INVALID && host_index < num_hosts_) {
                    out_ptr = host_base + host_index * num_feats_;
                    data_copy(out_ptr, in_ptr, num_feats_);
                }
            });

        return true;
    }

    void sync_update() {
        cudaStreamSynchronize(stream[2]);
    }

    friend bool
    check_part_compute_state(const GPUDataCache &aggr_cache, 
                             const GPUDataCache &emb_cache, 
                             const torch::Tensor &aggr_mapped_dst, 
                             const torch::Tensor &emb_src, 
                             const torch::Tensor &src_indices, 
                             const torch::Tensor &o_type, 
                             torch::Tensor &can_part_compute, // init with false
                             torch::Tensor &need_src_emb); // init with false
};

bool
check_part_compute_state(const GPUDataCache &aggr_cache, 
                         const GPUDataCache &emb_cache, 
                         const torch::Tensor &aggr_mapped_dst, 
                         const torch::Tensor &emb_src, 
                         const torch::Tensor &src_indices, 
                         const torch::Tensor &o_type, 
                         torch::Tensor &can_part_compute, // init with false
                         torch::Tensor &need_src_emb) { // init with false
    if (can_part_compute.size(0) != aggr_mapped_dst.size(0)) {
        return false;
    }

    const auto policy0 = thrust::cuda::par.on(aggr_cache.stream[0]);
    thrust::counting_iterator<int64_t> first(0);
    thrust::for_each(policy0, first, first + aggr_mapped_dst.size(0), 
        [INVALID = emb_cache.INVALID,
         emb_num_items_ = emb_cache.num_items,
         emb_num_hosts_ = emb_cache.num_host_items,
         emb_cache_index = emb_cache.device_cache_index,
         src_data = emb_src.data_ptr<int64_t>(),
         src_indices = src_indices.data_ptr<int64_t>(),
         o_type = o_type.data_ptr<int8_t>(),
         aggr_num_items_ = aggr_cache.num_items,
         aggr_num_hosts_ = aggr_cache.num_host_items,
         aggr_cache_index = aggr_cache.device_cache_index,
         dst_data = aggr_mapped_dst.data_ptr<int64_t>(),
         can_part = can_part_compute.data_ptr<bool>(), 
         need_src = need_src_emb.data_ptr<bool>()
         ] __device__ (int64_t i) {
            int64_t key = dst_data[i];
            int64_t device_index = aggr_cache_index[key].cache_index;
            int64_t host_index = aggr_cache_index[key].host_index;
            // dst must be in aggr cache (old aggr)
            if ((device_index > INVALID && device_index < aggr_num_items_) 
             || (host_index > INVALID && host_index < aggr_num_hosts_)) {
                bool res = true;
                // src must be in emb cache (old emb)
                for (int j = src_indices[i]; j < src_indices[i + 1]; j++) {
                    if (o_type[j] > 0) { // new emb
                        continue;
                    }
                    int64_t src_key = src_data[j];
                    int64_t src_device_index = emb_cache_index[src_key].cache_index;
                    int64_t src_host_index = emb_cache_index[src_key].host_index;
                    if ((src_device_index > INVALID && src_device_index < emb_num_items_) 
                     || (src_host_index > INVALID && src_host_index < emb_num_hosts_)) {
                        continue;
                    } else {
                        res = false;
                        break;
                    }
                }
                if (res) {
                    can_part[i] = true;
                    for (int j = src_indices[i]; j < src_indices[i + 1]; j++) {
                        need_src[j] = true;
                    }
                }
            }
        });

    return true;
}


namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<GPUDataCache>(m, "GPUDataCache")
        .def(py::init<size_t, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &, const torch::Tensor &, int, bool>())
        .def("query", &GPUDataCache::query)
        .def("check_state", &GPUDataCache::check_state)
        .def("invalidate", &GPUDataCache::invalidate)
        .def("replace", &GPUDataCache::replace) 
        // .def("replace_dev", &GPUDataCache::replace_dev)
        .def("replace_from_host", &GPUDataCache::replace_from_host)
        .def("update_hotness", &GPUDataCache::update_hotness)
        .def("sync_replace", &GPUDataCache::sync_replace)
        .def("update", &GPUDataCache::update)
        .def("sync_update", &GPUDataCache::sync_update);
    m.def("check_part_compute_state", &check_part_compute_state);
}

