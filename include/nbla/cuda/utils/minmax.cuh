// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NBLA_CUDA_UTILS_MINMAX_CUH__
#define __NBLA_CUDA_UTILS_MINMAX_CUH__

#include <nbla/cuda/limits.hpp>
#include <nbla/cuda/utils/warp_shuffle.cuh>
#include <nbla/cuda/utils/warp_vote.cuh>

namespace nbla {

template <typename T> struct MinMax {
  T min, max;

  __device__ __forceinline__ MinMax() {}

  __device__ __forceinline__ MinMax(T min, T max) : min(min), max(max) {}

  __device__ static MinMax<T> limits() {
    return MinMax<T>(+numeric_limits_cuda<T>::max(),
                     -numeric_limits_cuda<T>::max());
  }

  __device__ __forceinline__ void update(const MinMax<T> &other) {
    min = other.min < min ? other.min : min;
    max = other.max > max ? other.max : max;
  }

  __device__ __forceinline__ void update(const T &other) {
    min = other < min ? other : min;
    max = other > max ? other : max;
  }
};

template <> struct MinMax<float> {
  float min, max;

  __device__ __forceinline__ MinMax() {}

  __device__ __forceinline__ MinMax(float min, float max)
      : min(min), max(max) {}

  __device__ static MinMax<float> limits() {
    return MinMax<float>(+numeric_limits_cuda<float>::max(),
                         -numeric_limits_cuda<float>::max());
  }

  __device__ __forceinline__ void update(const MinMax<float> &other) {
    min = fminf(other.min, min);
    max = fmaxf(other.max, max);
  }

  __device__ __forceinline__ void update(const float &other) {
    min = fminf(other, min);
    max = fmaxf(other, max);
  }
};

namespace minmax_impl {

template <typename T>
__inline__ __device__ MinMax<T> warp_reduce(MinMax<T> val) {
  for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const MinMax<T> other(warp::shuffle_down(val.min, offset),
                          warp::shuffle_down(val.max, offset));
    val.update(other);
  }
  return val;
}

template <typename T>
__inline__ __device__ MinMax<T> block_reduce(MinMax<T> minmax) {
  static __shared__ MinMax<T> shared[1024];
  const unsigned int lane = threadIdx.x & CUDA_WARP_MASK;
  const unsigned int warp = threadIdx.x >> CUDA_WARP_BITS;

  minmax = warp_reduce(minmax);
  if (lane == 0)
    shared[warp] = minmax;
  __syncthreads();

  minmax = (threadIdx.x < blockDim.x / CUDA_WARP_SIZE) ? shared[lane]
                                                       : MinMax<T>::limits();

  if (warp == 0)
    minmax = warp_reduce(minmax);
  return minmax;
}

template <typename T, bool UseAbsVal = false>
__global__ void reduce(const T *data, const int size, MinMax<T> *minmax_data) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  MinMax<T> minmax = MinMax<T>::limits();

  for (int i = thread; i < size; i += stride) {
    minmax.update(UseAbsVal ? abs(data[i]) : data[i]);
  }

  minmax = block_reduce(minmax);

  if (threadIdx.x == 0) {
    minmax_data[blockIdx.x] = minmax;
  }
}

template <typename T, bool WipePartialResults = false>
__global__ void reduce(MinMax<T> *minmax_data, const int size) {
  const auto tid = threadIdx.x;
  MinMax<T> minmax = (tid < size) ? minmax_data[tid] : MinMax<T>::limits();

  minmax = block_reduce(minmax);

  // Write result to first element and wipe partials if requested.
  if (WipePartialResults) {
    minmax_data[tid] = (tid == 0) ? minmax : MinMax<T>(0, 0);
  } else if (tid == 0) {
    minmax_data[tid] = minmax;
  }
}

} // namespace minmax_impl

/*
  The minmax() function searches the `data` array of `size` elements
  for the minimum and maximum value and puts the result into the
  first item of `minmax_data`. The `minmax_data` array must be able
  to hold 1024 items that are used for intermediate reduction
  results, the final reduction will set all except the first
  `minmax_data` items to zero.
*/
template <typename T, bool UseAbsVal = false, bool WipePartialResults = false>
__host__ void minmax(const T *data, const int size, MinMax<T> *minmax_data) {
  using namespace minmax_impl;

  auto threads = NBLA_CUDA_NUM_THREADS;
  auto blocks = std::min(NBLA_CUDA_GET_BLOCKS(size), 1024);

  // Find min/max value per thread block, yields up to 1024 results.
  reduce<T, UseAbsVal><<<blocks, threads>>>(data, size, minmax_data);
  NBLA_CUDA_KERNEL_CHECK();

  // Find min/max value from the per-block results.
  reduce<T, WipePartialResults><<<1, 1024>>>(minmax_data, blocks);
  NBLA_CUDA_KERNEL_CHECK();
}

} // namespace nbla

#endif
