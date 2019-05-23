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

#ifndef __NBLA_CUDA_UTILS_TOP_K_CUH__
#define __NBLA_CUDA_UTILS_TOP_K_CUH__

#include <nbla/cuda/utils/bitonic_sort.cuh>
#include <nbla/cuda/utils/minmax.cuh>
#include <nbla/cuda/utils/warp_shuffle.cuh>

namespace nbla {

template <typename T> struct Bucket {
  T pivot;
  unsigned int count;
};

/*
  The bucket_count() kernel function searches the `data` array of
  `size` elements for values that are larger than the pivot value
  determined by the arithmetic mean of the minimum and maximum value
  given by `minmax`. The pivot and count are returned in `bucket`. The
  `bucket` count member must set to zero before calling this function
  unless the increment of count by the number of found elements is
  desired. Except for the first call (iter 0) the minmax search span
  is adjusted to the bucket that contains the k-th largest value. All
  threads atomically increment the bucket count. One thread saves the
  tested pivot value and copies the the minmax interval for the next
  iteration.
*/
template <bool UseAbsVal = false, typename T>
__global__ void bucket_count(const T *data, const int size,
                             const unsigned int K, const int iter,
                             MinMax<T> *minmax_data, Bucket<T> *bucket) {
  auto minmax = minmax_data[iter];

  if (minmax.max - minmax.min > 0) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (iter > 0) {
      if (bucket[iter - 1].count >= K)
        minmax.min = bucket[iter - 1].pivot;
      if (bucket[iter - 1].count <= K)
        minmax.max = bucket[iter - 1].pivot;
    }

    const T pivot = minmax.min + (minmax.max - minmax.min) * T(0.5);
    unsigned int count = 0;

    for (int i = thread; i < size; i += stride) {
      count += (UseAbsVal ? abs(data[i]) : data[i]) > pivot ? 1 : 0;
    }

    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset >>= 1)
      count += warp::shuffle_down(count, offset);

    if ((threadIdx.x & CUDA_WARP_MASK) == 0)
      atomicAdd(&bucket[iter].count, count);

    if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
      minmax_data[(iter + 1) & 31] = minmax;
      bucket[iter].pivot = pivot;
    }
  }
}

/*
  The bucket_reduce() kernel function searches the `bucket_data` array
  of 32 elements for the pivot value that gives the smallest number of
  k >= K values grater than the pivot value. All threads write the
  chosen pivot value and set count to zero to initialize for further
  use as atomic counter. This is a single warp kernel.
 */
template <typename T>
__global__ void bucket_reduce(const unsigned int K, Bucket<T> *bucket_data) {
  const auto laneid = threadIdx.x & CUDA_WARP_MASK;
  const auto bucket = bucket_data[laneid];
  unsigned int equal, count = bucket.count >= K ? bucket.count : 0xffffffff;

  for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset >>= 1)
    count = min(count, warp::shuffle_down(count, offset));

  count = warp::shuffle(count, 0); // broadcast
  equal = warp::ballot(count == bucket.count);
  bucket_data[laneid].pivot = bucket_data[__ffs(equal) - 1].pivot;
  bucket_data[laneid].count = 0; // clear for atomic counter
}

/*
  The find_top_k_value() function searches the `data` array of `size`
  elements for the `K`th, K <= 1024, largest value within the value
  range provided by `minmax`.

  The search is performed by 32 times splitting the `minmax` range
  with a pivot set to the half distance, counting the `data` elements
  larger than the pivot, and adjusting the `minmax` range to get
  closer to a value that yields K elements larger than the pivot. For
  randomly distributed `data` values the final pivot will yield
  exactly `K` elements with high probability. Convergence to exatly
  `K` elements may not be reached for highly repetitive `data` values,
  e.g. quantized or clipped values.

  The `bucket_data` pointer must reference a memory block that can
  accomodate 32 elements (32 * sizeof(Bucket<T>) bytes). Upon return,
  the first element of `bucket_data` provides the final pivot
  value. Note that the `bucket_data` count value is set to zero for
  later use as atomic counter.
 */
template <typename T, bool UseAbsVal = false>
__host__ void find_top_k_value(const T *data, const int size, MinMax<T> *minmax,
                               Bucket<T> *bucket_data, const unsigned int K) {
  auto threads = NBLA_CUDA_NUM_THREADS;
  auto blocks = NBLA_CUDA_GET_BLOCKS(size);

  for (int i = 0; i < CUDA_WARP_SIZE; i++) {
    // count values > min + 0.5 * (max - min)
    bucket_count<UseAbsVal><<<blocks, threads>>>(data, size, K, i, minmax,
                                                 bucket_data);
    NBLA_CUDA_KERNEL_CHECK();
  }

  bucket_reduce<<<1, CUDA_WARP_SIZE>>>(K, bucket_data);
  NBLA_CUDA_KERNEL_CHECK();
}

/*
  A value-index struct for the bitonic sorter to compare by value and
  shuffle both value and index between threads in a warp. The chosen
  compare() and extrema() method implementations yield a max to min
  sort output.
*/
template <typename T> struct ValIdx {
  T v;
  unsigned int i;

  __device__ __forceinline__ const T &value() const { return v; }
  __device__ __forceinline__ const unsigned int &index() const { return i; }

  static __device__ __forceinline__ ValIdx<T> extrema() {
    return {-numeric_limits_cuda<T>::max(), 0};
  }

  static __device__ __forceinline__ bool compare(const ValIdx<T> &a,
                                                 const ValIdx<T> &b) {
    return a.v > b.v;
  }

  static __device__ __forceinline__ ValIdx<T> shuffle(const ValIdx<T> var,
                                                      int mask) {
    return {warp::shuffle_xor(var.v, mask), warp::shuffle_xor(var.i, mask)};
  }
};

/*
  The init_val_idx_list() kernel function loads `data` values larger
  than the `bucket` pivot value and the corresponding index into
  subsequent elements of the `sort_data` array. Note that the `bucket`
  count is used as atomic counter and must initially be set to zero.
 */
template <typename T, bool UseAbsVal = false>
__global__ void init_val_idx_list(const T *data, const int size,
                                  Bucket<T> *bucket, ValIdx<T> *sort_data,
                                  const unsigned int sort_data_size) {
  const auto thread = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (unsigned int index = thread; index < size; index += stride) {
    T value = UseAbsVal ? abs(data[index]) : data[index];
    if (value > bucket->pivot) {
      sort_data[atomicInc(&bucket->count, sort_data_size)] = {value, index};
    }
  }
}

template <typename T, bool UseAbsVal = false>
__host__ void find_top_k_index(const T *data, const int size, Bucket<T> *bucket,
                               ValIdx<T> *sort_data, const unsigned int K) {
  auto threads = NBLA_CUDA_NUM_THREADS;
  auto blocks = NBLA_CUDA_GET_BLOCKS(size);

  init_val_idx_list<T, UseAbsVal><<<blocks, threads>>>(data, size, bucket,
                                                       sort_data, 1024);
  NBLA_CUDA_KERNEL_CHECK();

  bitonic_sort<<<1, 1024>>>(sort_data, K);
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T> struct Buffer {
  MinMax<T> minmax[32];
  Bucket<T> bucket[32];
  ValIdx<T> sorted[1024];
};

template <typename T, bool UseAbsVal = false>
__host__ void top_k(const T *data, const unsigned int size,
                    const unsigned int K, Buffer<T> *buffer) {
  minmax<T, UseAbsVal, true>(data, size, &buffer->minmax[0]);
  find_top_k_value<T, UseAbsVal>(data, size, &buffer->minmax[0],
                                 &buffer->bucket[0], K);
  find_top_k_index<T, UseAbsVal>(data, size, &buffer->bucket[0],
                                 &buffer->sorted[0], K);
}
} // namespace nbla

#endif
