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
#include <nbla/cuda/utils/warp_reduce.cuh>

namespace nbla {

template <typename index_t> struct WelfordType {
  float mean;
  float m2;
  index_t n;
};

template <>
__forceinline__ __device__ WelfordType<Size_t>
shuffle_down(WelfordType<Size_t> val, int offset, int width) {
  WelfordType<Size_t> buff;
  buff.mean = shuffle_down(val.mean, offset, width);
  buff.m2 = shuffle_down(val.m2, offset, width);
  buff.n = shuffle_down(val.n, offset, width);
  return buff;
}

template <typename Tcu, typename index_t> class WelfordOp {
public:
  using AccT = WelfordType<index_t>;
  using PreloadT = Tcu;
  // For the compatibility with `block_reduce.cuh` and `warp_reduce.cuh.`
  using storage_type = AccT;

private:
  // Input buffer
  const Tcu *x_;

  // Output buffer
  Tcu *mean_, *var_;

  const index_t reduce_size_;

public:
  WelfordOp(const Tcu *x, Tcu *mean, Tcu *var, const index_t reduce_size)
      : x_(x), mean_(mean), var_(var), reduce_size_(reduce_size) {}

  // Load
  __forceinline__ __device__ PreloadT load(const index_t idx, const index_t) {
    return x_[idx];
  }

  // Reduce one
  __forceinline__ __device__ void reduce_one(AccT &to, const PreloadT x) {
    const float dmean = x - to.mean;
    const float next_mean = to.mean + dmean / (to.n + 1);
    const float next_m2 = to.m2 + dmean * (x - next_mean);
    to.mean = next_mean;
    to.m2 = next_m2;
    to.n = to.n + 1;
  }

  // Initialize the initial value of reduction.
  __forceinline__ __device__ static void init(AccT &v) {
    v.mean = 0.0f;
    v.m2 = 0.0f;
    v.n = 0;
  }

  // Reduction of the two value
  __forceinline__ __device__ void reduce(storage_type &to,
                                         const storage_type &from) {
    if (to.n == 0) {
      to = from;
      return;
    }
    if (from.n == 0) {
      return;
    }
    const index_t next_n = to.n + from.n;
    const float next_n_inv = 1.0f / next_n;
    const float dmean = from.mean - to.mean;
    const float next_mean = to.mean + dmean * from.n * next_n_inv;
    const float next_m2 =
        to.m2 + from.m2 + dmean * dmean * to.n * from.n * next_n_inv;
    to.mean = next_mean;
    to.m2 = next_m2;
    to.n = next_n;
  }

  // Store the results
  __forceinline__ __device__ void store(const index_t idx, const AccT &v) {
    mean_[idx] = v.mean;
    var_[idx] = v.m2 / reduce_size_;
  }
};

// Reduction operator for GroupNormalization backward.
template <typename Tcu, typename index_t> class GNGradOp {
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
  __forceinline__ __device__ PreloadT load(const index_t idx, const index_t) {
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
  __forceinline__ __device__ void store(const index_t idx, const AccT &v) {
    sum_dy_[idx] = v.x;
    sum_dyx_[idx] = v.y;
  }
};

// InstanceNormalization can use a same operator for backward reduction.
template <typename Tcu, typename index_t>
using INGradOp = GNGradOp<Tcu, index_t>;

// Reduction operator for LayerNormalization backward.
template <typename Tcu, typename index_t> class LNGradOp {
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
  __forceinline__ __device__ PreloadT load(const index_t global_idx,
                                           const index_t reduce_idx) {
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
  __forceinline__ __device__ void store(const index_t idx, const AccT &v) {
    sum_dygamma_[idx] = v.x;
    sum_dyxgamma_[idx] = v.y;
  }
};

template <typename Op, typename index_t>
__global__ void reduce_2d_x(Op op, index_t reduce_size) {
  const index_t bidx = blockIdx.x;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  // Load and reduce
  typename Op::AccT val;
  Op::init(val);
  constexpr index_t B = 8;
  typename Op::PreloadT reg[B];
  for (index_t i = tidx; i < reduce_size; i += bdimx * B) {
#pragma unroll
    for (index_t j = 0; j < B; j++) {
      if (i + bdimx * j < reduce_size) {
        const index_t idx = bidx * reduce_size + i + bdimx * j;
        reg[j] = op.load(idx, i);
      }
    }
#pragma unroll
    for (index_t j = 0; j < B; j++) {
      if (i + bdimx * j < reduce_size) {
        op.reduce_one(val, reg[j]);
      }
    }
  }
  blockReduce(op, val);

  // Store
  if (threadIdx.x == 0) {
    op.store(bidx, val);
  }
}
}
