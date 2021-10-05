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

namespace nbla {

// Use custom maximum blocks constant because NBLA_CUDA_MAX_BLOCKS is 65536
// which does not match official CUDA document.
constexpr size_t NBLA_CUDA_LN_MAX_BLOCKS = 65535;
constexpr size_t NBLA_CUDA_LN_NUM_THREADS = NBLA_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void
layer_norm_forward_normalization(const IndexT outer_size,
                                 const IndexT reduce_size, const T *x,
                                 const T *mean, const T *var, const T *beta,
                                 const T *gamma, T *y, const float eps) {

  const IndexT bidy = blockIdx.y;
  const IndexT gdimy = gridDim.y;
  const IndexT tidx = threadIdx.x;
  const IndexT bdimx = blockDim.x;

  // Grid-stride loop
  for (IndexT outer_idx = blockIdx.x; outer_idx < outer_size;
       outer_idx += gridDim.x) {
    for (IndexT i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
      const IndexT idx = outer_idx * reduce_size + i;
      const T scale = gamma ? gamma[i] : (T)1.0f;
      const T bias = beta ? beta[i] : (T)0.0f;
      const T invstd = rsqrt(var[outer_idx] + eps);

      y[idx] = scale * invstd * (x[idx] - mean[outer_idx]) + bias;
    }
  }
}

template <typename T, typename IndexT>
__global__ void layer_norm_backward_dx_factor(
    const IndexT batch_size, const float inv_reduce_size, const T *mean,
    const T *var, const T *dmean, const T *dvar, const T *sum_dygamma,
    const T *sum_dyxgamma, T *factor_a, T *factor_b, const float eps) {

  // Grid-stride loop
  for (IndexT idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size;
       idx += gridDim.x * blockDim.x) {
    const float invstd = rsqrt(var[idx] + eps);

    const float tmp = (sum_dygamma[idx] * mean[idx] - sum_dyxgamma[idx]) *
                          invstd * invstd * invstd * inv_reduce_size +
                      (dvar ? 2.0f * dvar[idx] * inv_reduce_size : 0.0f);

    factor_a[idx] = tmp;
    factor_b[idx] = -tmp * mean[idx] -
                    sum_dygamma[idx] * invstd * inv_reduce_size +
                    (dmean ? dmean[idx] * inv_reduce_size : 0.0f);
  }
}

template <bool accum, typename T, typename IndexT>
__global__ void
layer_norm_backward_dx(const IndexT outer_size, const IndexT reduce_size,
                       const T *x, const T *dy, const T *gamma, const T *var,
                       const T *factor_a, const T *factor_b, T *dx,
                       const float eps) {
  const IndexT bidy = blockIdx.y;
  const IndexT gdimy = gridDim.y;
  const IndexT tidx = threadIdx.x;
  const IndexT bdimx = blockDim.x;

  // Grid-stride loop
  for (IndexT outer_idx = blockIdx.x; outer_idx < outer_size;
       outer_idx += gridDim.x) {
    for (IndexT i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
      const IndexT idx = outer_idx * reduce_size + i;
      const T scale = gamma ? gamma[i] : (T)1.0f;
      const T invstd = rsqrt(var[outer_idx] + eps);

      dx[idx] = dy[idx] * invstd * scale + factor_a[outer_idx] * x[idx] +
                factor_b[outer_idx] + (accum ? dx[idx] : (T)0.0f);
    }
  }
}

template <bool accum_beta, bool accum_gamma, typename T, typename IndexT>
__global__ void layer_norm_backward_dbeta_dgamma(
    const IndexT batch_size, const IndexT reduce_size, const T *x, const T *dy,
    const T *mean, const T *var, T *dbeta_out, T *dgamma_out, const float eps) {
  // Grid-stride loop
  for (IndexT idx = blockIdx.x * blockDim.x + threadIdx.x; idx < reduce_size;
       idx += gridDim.x * blockDim.x) {
    float dbeta = 0.0f;
    float dgamma = 0.0f;
    for (IndexT i = 0; i < batch_size; i++) {
      const IndexT global_idx = i * reduce_size + idx;
      const float invstd = rsqrt(var[i] + eps);
      dbeta += static_cast<float>(dy[global_idx]);
      dgamma += dy[global_idx] * (x[global_idx] - mean[i]) * invstd;
    }

    if (dbeta_out) {
      dbeta_out[idx] = dbeta + (accum_beta ? dbeta_out[idx] : (T)0.0f);
    }
    if (dgamma_out) {
      dgamma_out[idx] = dgamma + (accum_gamma ? dgamma_out[idx] : (T)0.0f);
    }
  }
}
}
