// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_CUDA_UTILS_DEVICE_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_DEVICE_REDUCE_CUH__

#include <nbla/cuda/utils/block_reduce.cuh>

#include <nbla/cuda/common.hpp>

namespace nbla {

/** Generic block-wise reduction kernel.

@param[in] N Number of valid input items.
@param[in,out] op Reduction operator class. TODO: doc.
 */
template <class ReduceOp>
__global__ void kernel_reduce_per_block(const int N, ReduceOp op,
                                        int offset_in = 0, int offset_out = 0) {
  typename ReduceOp::storage_type thread_data;
  op.init(thread_data);
  NBLA_CUDA_KERNEL_LOOP(i, N) {
    op.reduce(thread_data, op.premap(i + offset_in));
  }
  blockReduce(op, thread_data);
  if (threadIdx.x == 0) {
    op.postmap(blockIdx.x + offset_out, thread_data);
  }
}

template <class ReduceOp>
__global__ void kernel_reduce_2d_naive(const int outer_size,
                                       const int reduction_size, ReduceOp op) {
  typename ReduceOp::storage_type v;
  NBLA_CUDA_KERNEL_LOOP(o, outer_size) {
    op.init(v);
    // Reduction loop
    for (int i = 0; i < reduction_size; ++i) {
      op.reduce(v, op.premap(o * reduction_size + i));
    }
    op.postmap(o, v);
  }
}

template <class ReduceOp>
void reduce_2d_naive(const int outer_size, const int reduction_size,
                     ReduceOp op) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_2d_naive, outer_size,
                                 reduction_size, op);
}

/** Coalescing warp-strided 2D reduction.

    `kernel_reduce_2d_naive` is not coalescing memory access which is very
   inefficient. This does coalesced access by warp-strided inner loop followed
   by warp reduction by shuffle down. One warp is allocated for each outer axis.
 */
template <class ReduceOp>
__global__ void kernel_reduce_2d_mixed_parallel(const int outer_size,
                                                const int reduction_size,
                                                ReduceOp op) {
  typename ReduceOp::storage_type v;
  const int lane_id = threadIdx.x % warpSize; // Lane id in a warp.
  // Grid-strided loop (with warp)
  NBLA_CUDA_KERNEL_LOOP(idx, outer_size * warpSize) {
    const int o = idx / warpSize; // An axis for a warp.
    // Reduction loop (warp-strided loop)
    // The memory accesses are coalesced.
    op.init(v);
    for (int j = 0; j < NBLA_CEIL_INT_DIV(reduction_size, warpSize); ++j) {
      const int i = lane_id + j * warpSize;
      if (i < reduction_size) {
        op.reduce(v, op.premap(o * reduction_size + i));
      }
    }
    warpReduce(op, v);
    if (lane_id == 0) {
      op.postmap(o, v);
    }
  }
}

template <class ReduceOp>
void reduce_2d_mixed_parallel(const int outer_size, const int reduction_size,
                              ReduceOp op) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_2d_mixed_parallel, outer_size,
                                 reduction_size, op);
}

inline int cuda_get_reduction_blocks(int reduction_size) {
  return min(NBLA_CUDA_GET_BLOCKS(reduction_size), /*max blocks*/ 1024);
}

template <typename T>
std::pair<shared_ptr<CudaCachedArray>, T *>
cuda_get_reduction_buffer(int reduction_size, const Context &ctx) {
  const int blocks = cuda_get_reduction_blocks(reduction_size);
  shared_ptr<CudaCachedArray> arr_block =
      make_shared<CudaCachedArray>(blocks, get_dtype<T>(), ctx);
  T *block = arr_block->pointer<T>();
  return {arr_block, block};
}

template <class ReducePreOp, class ReducePostOp>
void reduce_2d_parallel_reduction(const int outer_size,
                                  const int reduction_size, ReducePreOp pre_op,
                                  ReducePostOp post_op) {
  // Get block size
  int blocks = cuda_get_reduction_blocks(reduction_size);

  // Per axis reduction
  for (int o = 0; o < outer_size; ++o) {
    kernel_reduce_per_block<<<blocks, NBLA_CUDA_NUM_THREADS>>>(
        reduction_size, pre_op, o * reduction_size);
    NBLA_CUDA_KERNEL_CHECK();
    kernel_reduce_per_block<<<1, 1024>>>(blocks, post_op, 0, o);
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}
#endif