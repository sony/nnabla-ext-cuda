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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/layer_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/cuda/utils/warp_reduce.cuh>

namespace nbla {

template <typename T>
void LayerNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  LayerNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto ndim = x->ndim();
  const auto x_size = x->size();
  const auto x_shape = x->shape();

  batch_size_ = 1;
  for (auto b : this->batch_axis_) {
    batch_size_ *= x_shape[b];
  }

  reduce_size_ = x_size / batch_size_;

  // Variable factor_a_, factor_b_, factor_c_;
  mean_.reshape({batch_size_}, true);
  var_.reshape({batch_size_}, true);
  sum_dygamma_.reshape({batch_size_}, true);
  sum_dyxgamma_.reshape({batch_size_}, true);

  factor_a_.reshape({batch_size_}, true);
  factor_b_.reshape({batch_size_}, true);
}

template <typename index_t> struct WelfordType {
  float mean;
  float m2;
  index_t n;
};

template <typename index_t> class WelfordOp {
public:
  using storage_type = WelfordType<index_t>;

  __forceinline__ __device__ void init(storage_type &thread_data) {
    thread_data.mean = 0.0f;
    thread_data.m2 = 0.0f;
    thread_data.n = 0;
  }

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

  __forceinline__ __device__ void reduce_one(storage_type &to, const float x) {
    const float dmean = x - to.mean;
    const float next_mean = to.mean + dmean / (to.n + 1);
    const float next_m2 = to.m2 + dmean * (x - next_mean);
    to.mean = next_mean;
    to.m2 = next_m2;
    to.n = to.n + 1;
  }
};

// Explicit template instantiation of shuffle_down for WelfordType
template <>
__forceinline__ __device__ WelfordType<int>
shuffle_down(WelfordType<int> val, int offset, int width) {
  WelfordType<int> buff;
  buff.mean = shuffle_down(val.mean, offset, width);
  buff.m2 = shuffle_down(val.m2, offset, width);
  buff.n = shuffle_down(val.n, offset, width);
  return buff;
}

template <>
__forceinline__ __device__ WelfordType<Size_t>
shuffle_down(WelfordType<Size_t> val, int offset, int width) {
  WelfordType<Size_t> buff;
  buff.mean = shuffle_down(val.mean, offset, width);
  buff.m2 = shuffle_down(val.m2, offset, width);
  buff.n = shuffle_down(val.n, offset, width);
  return buff;
}

template <typename T, typename index_t>
__global__ void layer_norm_forward_mean_var(const index_t reduce_size,
                                            const T *x, T *mean, T *var) {

  const index_t bidx = blockIdx.x;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  WelfordOp<index_t> op;

  // Load and reduce
  WelfordType<index_t> val;
  op.init(val);
  constexpr index_t B = 8;
  T reg[B];
  for (index_t i = tidx; i < reduce_size; i += bdimx * B) {
#pragma unroll
    for (index_t j = 0; j < B; j++) {
      if (i + bdimx * j < reduce_size) {
        const index_t idx = bidx * reduce_size + i + bdimx * j;
        reg[j] = x[idx];
      }
    }
#pragma unroll
    for (index_t j = 0; j < B; j++) {
      if (i + bdimx * j < reduce_size) {
        op.reduce_one(val, float(reg[j]));
      }
    }
  }

  blockReduce(op, val);

  // Store
  if (threadIdx.x == 0) {
    mean[bidx] = val.mean;
    var[bidx] = val.m2 / reduce_size;
  }
}

template <typename T, typename index_t>
__global__ void
layer_norm_forward_normalization(const index_t reduce_size, const T *x,
                                 const T *mean, const T *var, const T *beta,
                                 const T *gamma, T *y, const float eps) {

  const index_t bidx = blockIdx.x;
  const index_t bidy = blockIdx.y;
  const index_t gdimy = gridDim.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  for (index_t i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
    const index_t idx = bidx * reduce_size + i;
    const T scale = gamma ? gamma[i] : (T)1.0f;
    const T bias = beta ? beta[i] : (T)0.0f;
    const T invstd = rsqrt(var[bidx] + eps);

    y[idx] = scale * invstd * (x[idx] - mean[bidx]) + bias;
  }
}

template <typename T, typename index_t>
__global__ void layer_norm_backward_sum_dygamma_dyxgamma(
    const index_t reduce_size, const T *x, const T *gamma, const T *dy,
    T *sum_dygamma_out, T *sum_dyxgamma_out) {
  const index_t bidx = blockIdx.x;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  // Load and reduce
  float sum_dygamma = 0.0f;
  float sum_dyxgamma = 0.0f;
  for (index_t i = tidx; i < reduce_size; i += bdimx) {
    const index_t idx = bidx * reduce_size + i;
    const float scale = gamma ? static_cast<float>(gamma[i]) : 1.0f;
    sum_dygamma += static_cast<float>(dy[idx]) * scale;
    sum_dyxgamma +=
        static_cast<float>(dy[idx]) * static_cast<float>(x[idx]) * scale;
  }

  sum_dygamma = blockReduceSum(sum_dygamma);
  sum_dyxgamma = blockReduceSum(sum_dyxgamma);

  // Store
  if (threadIdx.x == 0) {
    sum_dygamma_out[bidx] = sum_dygamma;
    sum_dyxgamma_out[bidx] = sum_dyxgamma;
  }
}

template <typename T, typename index_t>
__global__ void layer_norm_backward_dx_factor(
    const index_t batch_size, const index_t reduce_size, const T *mean,
    const T *var, const T *dmean, const T *dvar, const T *sum_dygamma,
    const T *sum_dyxgamma, T *factor_a, T *factor_b, const float eps) {
  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    const float inv_reduce_size = 1.0f / reduce_size;
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

template <bool accum, typename T, typename index_t>
__global__ void
layer_norm_backward_dx(const index_t reduce_size, const T *x, const T *dy,
                       const T *gamma, const T *var, const T *factor_a,
                       const T *factor_b, T *dx, const float eps) {

  const index_t bidx = blockIdx.x;
  const index_t bidy = blockIdx.y;
  const index_t gdimy = gridDim.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  for (index_t i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
    const index_t idx = bidx * reduce_size + i;
    const T scale = gamma ? gamma[i] : (T)1.0f;
    const T invstd = rsqrt(var[bidx] + eps);

    if (accum) {
      dx[idx] +=
          dy[idx] * invstd * scale + factor_a[bidx] * x[idx] + factor_b[bidx];
    } else {
      dx[idx] =
          dy[idx] * invstd * scale + factor_a[bidx] * x[idx] + factor_b[bidx];
    }
  }
}

template <bool accum_beta, bool accum_gamma, typename T, typename index_t>
__global__ void
layer_norm_backward_dbeta_dgamma(const index_t batch_size,
                                 const index_t reduce_size, const T *x,
                                 const T *dy, const T *mean, const T *var,
                                 T *dbeta_out, T *dgamma_out, const float eps) {
  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < reduce_size) {
    float dbeta = 0.0f;
    float dgamma = 0.0f;
    for (index_t i = 0; i < batch_size; i++) {
      const index_t global_idx = i * reduce_size + idx;
      const float invstd = rsqrt(var[i] + eps);
      dbeta += static_cast<float>(dy[global_idx]);
      dgamma += dy[global_idx] * (x[global_idx] - mean[i]) * invstd;
    }

    if (dbeta_out) {
      if (accum_beta) {
        dbeta_out[idx] += dbeta;
      } else {
        dbeta_out[idx] = dbeta;
      }
    }
    if (dgamma_out) {
      if (accum_gamma) {
        dgamma_out[idx] += dgamma;
      } else {
        dgamma_out[idx] = dgamma;
      }
    }
  }
}

template <typename T>
void LayerNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);

  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate mean and variance
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid = batch_size_;
    const auto block = 512;

    layer_norm_forward_mean_var<<<grid, block>>>(reduce_size_, x, mean, var);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Layer normalization
  {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *beta = this->no_bias_
                         ? nullptr
                         : inputs[beta_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);

    // TODO:
    const dim3 grid(batch_size_,
                    NBLA_CEIL_SIZE_T_DIV(NBLA_CUDA_MAX_BLOCKS, batch_size_));
    const auto block = 512;

    layer_norm_forward_normalization<<<grid, block>>>(
        reduce_size_, x, mean, var, beta, gamma, y, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

template <typename T>
void LayerNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);

  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate sum of dy * gamma and sum of dy * x * gamma.
  {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dygamma = sum_dygamma_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyxgamma = sum_dyxgamma_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid = batch_size_;
    const auto block = 512;

    layer_norm_backward_sum_dygamma_dyxgamma<<<grid, block>>>(
        reduce_size_, x, gamma, dy, sum_dygamma, sum_dyxgamma);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate a and b such that `dx = gamma / sqrt(var) * dy + a * x + b`.
  {
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *dmean = outputs.size() == 3
                          ? v_mean->get_grad_pointer<Tc>(this->ctx_)
                          : nullptr;
    const Tc *dvar =
        outputs.size() == 3 ? v_var->get_grad_pointer<Tc>(this->ctx_) : nullptr;
    const Tc *sum_dygamma = sum_dygamma_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyxgamma = sum_dyxgamma_.get_data_pointer<Tc>(this->ctx_);

    Tc *factor_a = factor_a_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *factor_b = factor_b_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid = NBLA_CEIL_SIZE_T_DIV(batch_size_, 256);
    const auto block = 256;

    layer_norm_backward_dx_factor<<<grid, block>>>(
        batch_size_, reduce_size_, mean, var, dmean, dvar, sum_dygamma,
        sum_dyxgamma, factor_a, factor_b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx.
  if (propagate_down[0]) {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *factor_a = factor_a_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor_b = factor_b_.get_data_pointer<Tc>(this->ctx_);

    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);

    // TODO:
    const dim3 grid(batch_size_,
                    NBLA_CEIL_SIZE_T_DIV(NBLA_CUDA_MAX_BLOCKS, batch_size_));
    const auto block = 512;

    if (accum[0]) {
      layer_norm_backward_dx<true><<<grid, block>>>(
          reduce_size_, x, dy, gamma, var, factor_a, factor_b, dx, this->eps_);
    } else {
      layer_norm_backward_dx<false><<<grid, block>>>(
          reduce_size_, x, dy, gamma, var, factor_a, factor_b, dx, this->eps_);
    }
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dbeta and dgamma.
  if ((inputs.size() > 1 && propagate_down[1]) ||
      (inputs.size() > 2 && propagate_down[2])) {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    Tc *dbeta = !this->no_bias_ && propagate_down[beta_idx]
                    ? inputs[beta_idx]->cast_grad_and_get_pointer<Tc>(
                          this->ctx_, !accum[beta_idx])
                    : nullptr;
    Tc *dgamma = !this->no_scale_ && propagate_down[gamma_idx]
                     ? inputs[gamma_idx]->cast_grad_and_get_pointer<Tc>(
                           this->ctx_, !accum[gamma_idx])
                     : nullptr;

    const auto grid = NBLA_CEIL_SIZE_T_DIV(reduce_size_, 256);
    const auto block = 256;

    if (!this->no_bias_ && accum[beta_idx]) {
      if (!this->no_scale_ && accum[gamma_idx]) {
        layer_norm_backward_dbeta_dgamma<true, true><<<grid, block>>>(
            batch_size_, reduce_size_, x, dy, mean, var, dbeta, dgamma,
            this->eps_);
      } else {
        layer_norm_backward_dbeta_dgamma<true, false><<<grid, block>>>(
            batch_size_, reduce_size_, x, dy, mean, var, dbeta, dgamma,
            this->eps_);
      }
    } else {
      if (!this->no_scale_ && accum[gamma_idx]) {
        layer_norm_backward_dbeta_dgamma<false, true><<<grid, block>>>(
            batch_size_, reduce_size_, x, dy, mean, var, dbeta, dgamma,
            this->eps_);
      } else {
        layer_norm_backward_dbeta_dgamma<false, false><<<grid, block>>>(
            batch_size_, reduce_size_, x, dy, mean, var, dbeta, dgamma,
            this->eps_);
      }
    }
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}
