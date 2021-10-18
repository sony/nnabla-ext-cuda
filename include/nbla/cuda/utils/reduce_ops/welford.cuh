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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_WELFORD_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_WELFORD_CUH__

#include <nbla/cuda/utils/types.cuh>

namespace nbla {

/**
 * @brief Reduction operator to compute mean and variance.
 *
 * This operator uses parallel version of Welford's algorithm [0] for
 * calculating mean and variance.
 * [0] Updating Formulae and a Pairwise Algorithm for Computing Sample
 * Variances, Tony F. Chan, Gene H. Golub and Randall J. LeVeque, 1979,
 * Technical Report STAN-CS-79-773, Department of Computer Science, Stanford
 * University
 *
 * @tparam Tcu The type of the input and output values.
 * @tparam IndexT The type of the size, shape, and indices of the input and
 * output,
 */
template <typename Tcu, typename IndexT> class WelfordOp {
public:
  using AccT = WelfordType<IndexT>;
  using PreloadT = Tcu;
  // For the compatibility with `block_reduce.cuh` and `warp_reduce.cuh.`
  using storage_type = AccT;

private:
  // Input buffer
  const Tcu *x_;

  // Output buffer
  Tcu *mean_, *var_;

  const IndexT reduce_size_;

public:
  WelfordOp(const Tcu *x, Tcu *mean, Tcu *var, const IndexT reduce_size)
      : x_(x), mean_(mean), var_(var), reduce_size_(reduce_size) {}

  // Load
  __forceinline__ __device__ PreloadT load(const IndexT idx, const IndexT) {
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
    const IndexT next_n = to.n + from.n;
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
  __forceinline__ __device__ void store(const IndexT idx, const AccT &v) {
    mean_[idx] = v.mean;
    var_[idx] = v.m2 / reduce_size_;
  }
};
}
#endif