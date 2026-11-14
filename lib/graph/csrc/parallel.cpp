// Native threads read existing tensor/CSR/CSC storage directly. No Python API.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

template <class Function>
static int parallel_rows(std::int64_t count, std::int64_t threads, Function function) {
    if (count == 0) return 0;
    threads = std::max<std::int64_t>(1, std::min(threads, count));
    if (threads == 1) {
        for (std::int64_t i = 0; i < count; ++i) function(i);
        return 0;
    }
    std::atomic<std::int64_t> next{0};
    std::vector<std::thread> workers;
    try {
        workers.reserve(threads);
        for (std::int64_t thread = 0; thread < threads; ++thread) {
            workers.emplace_back([&] {
                while (true) {
                    const auto start = next.fetch_add(8, std::memory_order_relaxed);
                    if (start >= count) break;
                    const auto end = std::min(start + 8, count);
                    for (auto i = start; i < end; ++i) function(i);
                }
            });
        }
    } catch (...) {
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        return -1;
    }
    for (auto& worker : workers) worker.join();
    return 0;
}

extern "C" int frontier_expand_parallel(
    const std::int64_t* offsets, const std::int64_t* neighbors,
    const std::int64_t* nodes, std::int64_t count, bool* mask,
    std::int64_t* sources, std::int64_t* destinations, std::int64_t threads) {
    try {
        std::vector<std::int64_t> prefix(count + 1, 0);
        for (std::int64_t i = 0; i < count; ++i)
            prefix[i + 1] = prefix[i] + offsets[nodes[i] + 1] - offsets[nodes[i]];
        // Output ranges never overlap. Destination marks use atomic byte stores.
        return parallel_rows(count, threads, [&](std::int64_t i) {
            const auto source = nodes[i];
            auto out = prefix[i];
            for (auto edge = offsets[source]; edge < offsets[source + 1]; ++edge) {
                const auto destination = neighbors[edge];
                sources[out] = source;
                destinations[out++] = destination;
                __atomic_store_n(mask + destination, true, __ATOMIC_RELAXED);
            }
        });
    } catch (...) { return -2; }
}

extern "C" int pull_feature_delta(
    const std::int64_t* offsets, const std::int64_t* sources,
    const std::int64_t* targets, std::int64_t count, const std::int64_t* changed_rows,
    const float* current, std::int64_t current_row_stride, std::int64_t current_col_stride,
    const float* previous, std::int64_t previous_row_stride, std::int64_t previous_col_stride,
    double* aggregates, std::int64_t dimensions, std::int64_t threads) {
    try {
        // Each destination row has one owner; no atomic floating-point updates.
        return parallel_rows(count, threads, [&](std::int64_t i) {
            const auto destination = targets[i];
            auto* output = aggregates + destination * dimensions;
            for (auto edge = offsets[destination]; edge < offsets[destination + 1]; ++edge) {
                const auto source = sources[edge];
                const auto old_row = changed_rows[source];
                if (old_row < 0) continue;
                for (std::int64_t dim = 0; dim < dimensions; ++dim) {
                    output[dim] += static_cast<double>(current[source * current_row_stride + dim * current_col_stride])
                        - static_cast<double>(previous[old_row * previous_row_stride + dim * previous_col_stride]);
                }
            }
        });
    } catch (...) { return -2; }
}

extern "C" int frontier_mark_parallel(
    const std::int64_t* offsets, const std::int64_t* neighbors,
    const std::int64_t* nodes, std::int64_t count, bool* mask,
    std::int64_t threads) {
    try {
        // Borrow CSR and node buffers. Repeated destinations (including parallel
        // edges) are idempotent atomic stores, with no per-edge output arrays.
        return parallel_rows(count, threads, [&](std::int64_t i) {
            const auto source = nodes[i];
            for (auto edge = offsets[source]; edge < offsets[source + 1]; ++edge)
                __atomic_store_n(mask + neighbors[edge], true, __ATOMIC_RELAXED);
        });
    } catch (...) { return -2; }
}
