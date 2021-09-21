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

// Common kernels
#include <nbla/cuda/function/kernel/normalization.cuh>

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

  //----------------
  // Reshape buffers
  //----------------

  // Batch stats
  mean_.reshape({batch_size_}, true);
  var_.reshape({batch_size_}, true);

  // Internal buffers for backward calculation
  sum_dygamma_.reshape({batch_size_}, true);
  sum_dyxgamma_.reshape({batch_size_}, true);
  factor_a_.reshape({batch_size_}, true);
  factor_b_.reshape({batch_size_}, true);
}

template <typename T, typename index_t>
__global__ void
layer_norm_forward_normalization(const index_t outer_size,
                                 const index_t reduce_size, const T *x,
                                 const T *mean, const T *var, const T *beta,
                                 const T *gamma, T *y, const float eps) {

  const index_t bidy = blockIdx.y;
  const index_t gdimy = gridDim.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  // Grid-stride loop
  for (index_t outer_idx = blockIdx.x; outer_idx < outer_size;
       outer_idx += gridDim.x) {
    for (index_t i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
      const index_t idx = outer_idx * reduce_size + i;
      const T scale = gamma ? gamma[i] : (T)1.0f;
      const T bias = beta ? beta[i] : (T)0.0f;
      const T invstd = rsqrt(var[outer_idx] + eps);

      y[idx] = scale * invstd * (x[idx] - mean[outer_idx]) + bias;
    }
  }
}

template <typename T, typename index_t>
__global__ void layer_norm_backward_dx_factor(
    const index_t batch_size, const index_t reduce_size, const T *mean,
    const T *var, const T *dmean, const T *dvar, const T *sum_dygamma,
    const T *sum_dyxgamma, T *factor_a, T *factor_b, const float eps) {

  // Grid-stride loop
  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch_size;
       idx += gridDim.x * blockDim.x) {
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
layer_norm_backward_dx(const index_t outer_size, const index_t reduce_size,
                       const T *x, const T *dy, const T *gamma, const T *var,
                       const T *factor_a, const T *factor_b, T *dx,
                       const float eps) {
  const index_t bidy = blockIdx.y;
  const index_t gdimy = gridDim.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  // Grid-stride loop
  for (index_t outer_idx = blockIdx.x; outer_idx < outer_size;
       outer_idx += gridDim.x) {
    for (index_t i = tidx + bdimx * bidy; i < reduce_size; i += bdimx * gdimy) {
      const index_t idx = outer_idx * reduce_size + i;
      const T scale = gamma ? gamma[i] : (T)1.0f;
      const T invstd = rsqrt(var[outer_idx] + eps);

      if (accum) {
        dx[idx] += dy[idx] * invstd * scale + factor_a[outer_idx] * x[idx] +
                   factor_b[outer_idx];
      } else {
        dx[idx] = dy[idx] * invstd * scale + factor_a[outer_idx] * x[idx] +
                  factor_b[outer_idx];
      }
    }
  }
}

template <bool accum_beta, bool accum_gamma, typename T, typename index_t>
__global__ void
layer_norm_backward_dbeta_dgamma(const index_t batch_size,
                                 const index_t reduce_size, const T *x,
                                 const T *dy, const T *mean, const T *var,
                                 T *dbeta_out, T *dgamma_out, const float eps) {
  // Grid-stride loop
  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < reduce_size;
       idx += gridDim.x * blockDim.x) {
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

constexpr size_t LN_NUM_THREADS = NBLA_CUDA_NUM_THREADS;
constexpr size_t LN_MAX_BLOCKS = NBLA_CUDA_MAX_BLOCKS;

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

    const auto grid = std::min(batch_size_, static_cast<Size_t>(LN_MAX_BLOCKS));
    const auto block = LN_NUM_THREADS;

    WelfordOp<Tc, Size_t> op(x, mean, var, reduce_size_);
    reduce_2d_x<<<grid, block>>>(op, batch_size_, reduce_size_);
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

    const size_t elements_per_grid_y = LN_NUM_THREADS * 4;
    dim3 grid;
    grid.x = std::min(batch_size_, static_cast<Size_t>(LN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(LN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = LN_NUM_THREADS;

    layer_norm_forward_normalization<<<grid, block>>>(
        batch_size_, reduce_size_, x, mean, var, beta, gamma, y, this->eps_);
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
  if (propagate_down[0]) {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dygamma = sum_dygamma_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyxgamma = sum_dyxgamma_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid = std::min(batch_size_, static_cast<Size_t>(LN_MAX_BLOCKS));
    const auto block = LN_NUM_THREADS;

    LNGradOp<Tc, Size_t> op(x, gamma, dy, sum_dygamma, sum_dyxgamma);
    reduce_2d_x<<<grid, block>>>(op, batch_size_, reduce_size_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate a and b such that `dx = gamma / sqrt(var) * dy + a * x + b`.
  if (propagate_down[0]) {
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

    const auto grid = std::min(
        static_cast<Size_t>(LN_MAX_BLOCKS),
        static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(batch_size_, LN_NUM_THREADS)));
    const auto block = LN_NUM_THREADS;

    layer_norm_backward_dx_factor<<<grid, block>>>(
        batch_size_, reduce_size_, mean, var, dmean, dvar, sum_dygamma,
        sum_dyxgamma, factor_a, factor_b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();

    // Clear internal buffers
    sum_dygamma_.data()->array()->clear();
    sum_dyxgamma_.data()->array()->clear();
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

    const size_t elements_per_grid_y = LN_NUM_THREADS * 4;
    dim3 grid;
    grid.x = std::min(batch_size_, static_cast<Size_t>(LN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(LN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = LN_NUM_THREADS;

    if (accum[0]) {
      layer_norm_backward_dx<true><<<grid, block>>>(batch_size_, reduce_size_,
                                                    x, dy, gamma, var, factor_a,
                                                    factor_b, dx, this->eps_);
    } else {
      layer_norm_backward_dx<false><<<grid, block>>>(
          batch_size_, reduce_size_, x, dy, gamma, var, factor_a, factor_b, dx,
          this->eps_);
    }
    NBLA_CUDA_KERNEL_CHECK();

    // Clear internal buffers
    factor_a_.data()->array()->clear();
    factor_b_.data()->array()->clear();
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

    const auto grid = std::min(static_cast<Size_t>(LN_MAX_BLOCKS),
                               static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(
                                   reduce_size_, LN_NUM_THREADS)));
    const auto block = LN_NUM_THREADS;

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
