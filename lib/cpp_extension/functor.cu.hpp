#pragma once
#include <thrust/tuple.h>

template <typename T>
class value_at
{
    const T *x;

  public:
    value_at(const T *x) : x(x) {}

    __host__ __device__ T operator()(size_t i) const { return x[i]; }
};

template <size_t i>
struct thrust_get {
    template <typename T>
    __device__ auto operator()(const T &t) const
    {
        return thrust::get<i>(t);
    }
};
