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

#ifndef __NBLA_CUDA_UTILS_WARP_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_WARP_REDUCE_CUH__

#include <nbla/cuda/utils/shuffle_down.cuh>

namespace nbla {

template <class ReduceOp>
__forceinline__ __device__ void
shuffle_down_reduce(ReduceOp &op, typename ReduceOp::storage_type &val,
                    int offset, int width = 32) {
  const typename ReduceOp::storage_type buff = shuffle_down(val, offset, width);
  op.reduce(val, buff);
}

template <class ReduceOp>
__inline__ __device__ void warpReduce(ReduceOp &op,
                                      typename ReduceOp::storage_type &val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    shuffle_down_reduce(op, val, offset);
  }
}

template <typename T> __inline__ __device__ T warpReduceSum(T val) {
#if __CUDA_ARCH__ >= 300
#define SHFL_DOWN __shfl_down
#else
#define SHFL_DOWN pre_fermi_shfl_down
#endif
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += SHFL_DOWN(val, offset);
  }
  return val;
}

__inline__ __device__ float2 warpReduceSumOfFloat2(float2 val) {
#if __CUDA_ARCH__ >= 300
#define SHFL_DOWN __shfl_down
#else
#define SHFL_DOWN pre_fermi_shfl_down
#endif
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += SHFL_DOWN(val.x, offset);
    val.y += SHFL_DOWN(val.y, offset);
  }
  return val;
}

__inline__ __device__ float3 warpReduceSumOfFloat3(float3 val) {
#if __CUDA_ARCH__ >= 300
#define SHFL_DOWN __shfl_down
#else
#define SHFL_DOWN pre_fermi_shfl_down
#endif
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += SHFL_DOWN(val.x, offset);
    val.y += SHFL_DOWN(val.y, offset);
    val.z += SHFL_DOWN(val.z, offset);
  }
  return val;
}

__inline__ __device__ float4 warpReduceSumOfFloat4(float4 val) {
#if __CUDA_ARCH__ >= 300
#define SHFL_DOWN __shfl_down
#else
#define SHFL_DOWN pre_fermi_shfl_down
#endif
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += SHFL_DOWN(val.x, offset);
    val.y += SHFL_DOWN(val.y, offset);
    val.z += SHFL_DOWN(val.z, offset);
    val.w += SHFL_DOWN(val.w, offset);
  }
  return val;
}
}
#endif