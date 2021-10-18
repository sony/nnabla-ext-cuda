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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_GROUP_NORMLIZATION_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_GROUP_NORMLIZATION_CUH__

namespace nbla {

// Reduction operator for GroupNormalization backward.
template <typename Tcu, typename IndexT> class GNGradOp {
public:
  using AccT = float2;     // sum of dy and dy*x
  using PreloadT = float2; // x, dy
  // For the compatibility with `block_reduce.cuh` and `warp_reduce.cuh.`
  using storage_type = AccT;

private:
  // Input buffer
  const Tcu *x_, *dy_;

  // Output buffer
  Tcu *sum_dy_, *sum_dyx_;

public:
  GNGradOp(const Tcu *x, const Tcu *dy, Tcu *sum_dy, Tcu *sum_dyx)
      : x_(x), dy_(dy), sum_dy_(sum_dy), sum_dyx_(sum_dyx) {}

  // Load
  __forceinline__ __device__ PreloadT load(const IndexT idx, const IndexT) {
    return make_float2(x_[idx], dy_[idx]);
  }

  // Reduce one
  __forceinline__ __device__ static void reduce_one(AccT &to,
                                                    const PreloadT &v) {
    to.x += v.y;       // Sum of dy
    to.y += v.x * v.y; // Sum of dy*x
  }

  // Initialize the initial value of reduction.
  __forceinline__ __device__ static void init(AccT &v) {
    v.x = 0.0f;
    v.y = 0.0f;
  }

  // Reduction of the two value
  __forceinline__ __device__ static void reduce(storage_type &to,
                                                const storage_type &from) {
    to.x += from.x; // Sum of dy
    to.y += from.y; // Sum of dy*x
  }

  // Store the results
  __forceinline__ __device__ void store(const IndexT idx, const AccT &v) {
    sum_dy_[idx] = v.x;
    sum_dyx_[idx] = v.y;
  }
};
}
#endif