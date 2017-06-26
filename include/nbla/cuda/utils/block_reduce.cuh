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

#ifndef __NBLA_CUDA_UTILS_BLOCK_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_BLOCK_REDUCE_CUH__
#include <nbla/cuda/utils/warp_reduce.cuh>

#define SHARED_MEMORY_SIZE_PER_BLOCK 32

#define REDUCTION_NUM_BLOCKS 512

#define reduction_blocks(blocks, num)                                          \
  int blocks = min((N + NBLA_CUDA_NUM_THREADS - 1) / NBLA_CUDA_NUM_THREADS,    \
                   REDUCTION_NUM_BLOCKS)

namespace nbla {

/** Generic block reduce device function.

@param[in] op Reduction operator class. TODO: doc.
@param[in,out] val per-thread register storage for reduction.

 */
template <class ReduceOp>
__inline__ __device__ void blockReduce(ReduceOp &op,
                                       typename ReduceOp::storage_type &val) {
  static __shared__ typename ReduceOp::storage_type
      shared[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  warpReduce(op, val); // Each warp performs partial reduction
  if (lane == 0) {
    shared[wid] = val;
  }                // Write reduced value to shared memory
  __syncthreads(); // Wait for all partial reductions
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared[lane];
  } else {
    op.init(val);
  }
  if (wid == 0) {
    warpReduce(op, val); // Final reduce within first warp
  }
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {

  static __shared__ float
      shared[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val); // Each warp performs partial reduction
  if (lane == 0) {
    shared[wid] = val;
  }                // Write reduced value to shared memory
  __syncthreads(); // Wait for all partial reductions
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared[lane];
  } else {
    val = 0;
  }
  if (wid == 0) {
    val = warpReduceSum(val); // Final reduce within first warp
  }
  return val;
}

__inline__ __device__ float2 blockReduceSumOfFloat2(float2 val) {
  static __shared__ float shared_val1[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared
                                                                     // mem for
                                                                     // 32
                                                                     // partial
                                                                     // sums
  static __shared__ float shared_val2[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared
                                                                     // mem for
                                                                     // 32
                                                                     // partial
                                                                     // sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumOfFloat2(val); // Each warp performs partial reduction
  if (lane == 0) {
    shared_val1[wid] = val.x;
    shared_val2[wid] = val.y;
  }                // Write reduced value to shared memory
  __syncthreads(); // Wait for all partial reductions
  if (threadIdx.x < blockDim.x / warpSize) {
    val.x = shared_val1[lane];
    val.y = shared_val2[lane];
  } else {
    val.x = 0;
    val.y = 0;
  }
  if (wid == 0) {
    val = warpReduceSumOfFloat2(val); // Final reduce within first warp
  }
  return val;
}

__inline__ __device__ float3 blockReduceSumOfFloat3(float3 val) {

  static __shared__ float shared_val1[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared
                                                                     // mem for
                                                                     // 32
                                                                     // partial
                                                                     // sums
  static __shared__ float shared_val2[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared
                                                                     // mem for
                                                                     // 32
                                                                     // partial
                                                                     // sums
  static __shared__ float shared_val3[SHARED_MEMORY_SIZE_PER_BLOCK]; // Shared
                                                                     // mem for
                                                                     // 32
                                                                     // partial
                                                                     // sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumOfFloat3(val); // Each warp performs partial reduction
  if (lane == 0) {
    shared_val1[wid] = val.x;
    shared_val2[wid] = val.y;
    shared_val3[wid] = val.z;
  }                // Write reduced value to shared memory
  __syncthreads(); // Wait for all partial reductions
  if (threadIdx.x < blockDim.x / warpSize) {
    val.x = shared_val1[lane];
    val.y = shared_val2[lane];
    val.z = shared_val3[lane];
  } else {
    val.x = 0;
    val.y = 0;
    val.z = 0;
  }
  if (wid == 0) {
    val = warpReduceSumOfFloat3(val); // Final reduce within first warp
  }
  return val;
}
}
#endif