// Copyright 2020,2021 Sony Corporation.
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

#ifndef __NBLA_CUDA_UTILS_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_CUH__

#include <nbla/cuda/half.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>

namespace nbla {
template <typename T>
__global__ void kernel_reduce_per_block(const int N, const T *x, T *buff) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += (AccT)x[i]; }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    buff[blockIdx.x] = thread_data;
  }
}

template <typename T>
__global__ void kernel_reduce_xy_per_block(const int N, const T *x, const T *y,
                                           T *buff) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += ((AccT)x[i] * (AccT)y[i]); }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    buff[blockIdx.x] = thread_data;
  }
}

template <typename T>
__host__ void sum_prod(const Context &ctx, const T *x, const T *y, T *sum_xy,
                       int outer_size, int reduction_size) {
  if (reduction_size >= 1024) {
    const int threads = NBLA_CUDA_NUM_THREADS;
    const int blocks = min(NBLA_CUDA_GET_BLOCKS(reduction_size), 1024);
    shared_ptr<CudaCachedArray> arr_buff =
        make_shared<CudaCachedArray>(blocks, get_dtype<T>(), ctx);
    T *buff = arr_buff->pointer<T>();
    while (outer_size--) {
      kernel_reduce_xy_per_block<<<blocks, threads>>>(reduction_size, x, y,
                                                      buff);
      NBLA_CUDA_KERNEL_CHECK();
      kernel_reduce_per_block<<<1, 1024>>>(blocks, buff, sum_xy);
      NBLA_CUDA_KERNEL_CHECK();
      x += reduction_size;
      y += reduction_size;
      sum_xy += 1;
    }
  } else {
    while (outer_size--) {
      kernel_reduce_xy_per_block<<<1, 1024>>>(reduction_size, x, y, sum_xy);
      NBLA_CUDA_KERNEL_CHECK();
      x += reduction_size;
      y += reduction_size;
      sum_xy += 1;
    }
  }
}
} // namespace nbla

#endif