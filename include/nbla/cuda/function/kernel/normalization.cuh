// Copyright 2021 Sony Group Corporation.
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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/cuda/utils/types.cuh>
#include <nbla/cuda/utils/warp_reduce.cuh>

namespace nbla {

template <typename Op, typename IndexT>
__global__ void reduce_2d_x(Op op, IndexT outer_size, IndexT reduce_size) {
  const IndexT tidx = threadIdx.x;
  const IndexT bdimx = blockDim.x;

  // Grid-stride loop
  for (IndexT outer_idx = blockIdx.x; outer_idx < outer_size;
       outer_idx += gridDim.x) {

    // Load and reduce
    typename Op::AccT val;
    Op::init(val);
    constexpr IndexT B = 8;
    typename Op::PreloadT reg[B];
    for (IndexT i = tidx; i < reduce_size; i += bdimx * B) {
#pragma unroll
      for (IndexT j = 0; j < B; j++) {
        if (i + bdimx * j < reduce_size) {
          const IndexT global_idx = outer_idx * reduce_size + i + bdimx * j;
          const IndexT reduce_idx = i + bdimx * j;
          reg[j] = op.load(global_idx, reduce_idx);
        }
      }
#pragma unroll
      for (IndexT j = 0; j < B; j++) {
        if (i + bdimx * j < reduce_size) {
          op.reduce_one(val, reg[j]);
        }
      }
    }

    // Block reduce
    if (bdimx <= CUDA_WARP_SIZE) {
      warpReduce(op, val);
    } else {
      blockReduce(op, val);
    }

    // Store
    if (threadIdx.x == 0) {
      op.store(outer_idx, val);
    }
  }
}
} // namespace nbla
