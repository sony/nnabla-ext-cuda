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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_LAYER_NORMLIZATION_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_LAYER_NORMLIZATION_CUH__

namespace nbla {

// Reduction operator for LayerNormalization backward.
template <typename Tcu, typename IndexT> class LNGradOp {
public:
  using AccT = float2;     // sum of dy*gamma and dy*x*gamma
  using PreloadT = float3; // x, gamma, dy
  // For the compatibility with `block_reduce.cuh` and `warp_reduce.cuh.`
  using storage_type = AccT;

private:
  // Input buffer
  const Tcu *x_, *gamma_, *dy_;

  // Output buffer
  Tcu *sum_dygamma_, *sum_dyxgamma_;

public:
  LNGradOp(const Tcu *x, const Tcu *gamma, const Tcu *dy, Tcu *sum_dygamma,
           Tcu *sum_dyxgamma)
      : x_(x), gamma_(gamma), dy_(dy), sum_dygamma_(sum_dygamma),
        sum_dyxgamma_(sum_dyxgamma) {}

  // Load
  __forceinline__ __device__ PreloadT load(const IndexT global_idx,
                                           const IndexT reduce_idx) {
    const float gamma = gamma_ ? static_cast<float>(gamma_[reduce_idx]) : 1.0f;
    return make_float3(x_[global_idx], gamma, dy_[global_idx]);
  }

  // Reduce one
  __forceinline__ __device__ static void reduce_one(AccT &to,
                                                    const PreloadT &v) {
    to.x += v.z * v.y;       // Sum of dygamma
    to.y += v.z * v.x * v.y; // Sum of dy*x*gamma
  }

  // Initialize the initial value of reduction.
  __forceinline__ __device__ static void init(AccT &v) {
    v.x = 0.0f;
    v.y = 0.0f;
  }

  // Reduction of the two value
  __forceinline__ __device__ static void reduce(storage_type &to,
                                                const storage_type &from) {
    to.x += from.x; // Sum of dy*gamma
    to.y += from.y; // Sum of dy*x*gamma
  }

  // Store the results
  __forceinline__ __device__ void store(const IndexT idx, const AccT &v) {
    sum_dygamma_[idx] = v.x;
    sum_dyxgamma_[idx] = v.y;
  }
};
} // namespace nbla
#endif