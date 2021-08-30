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
#include <nbla/cuda/function/group_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/cuda/utils/warp_reduce.cuh>

// Common kernels
#include <nbla/cuda/function/kernel/normalization.cuh>

namespace nbla {

template <typename T>
void GroupNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  GroupNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto x_shape = x->shape();
  const auto ndim = x->ndim();
  const auto c = this->channel_axis_;
  channel_size_ = x_shape[c];

  need_adaptor_ = ChannelFirstAdaptor::need_adaptor(
      inputs[0]->shape(), this->batch_axis_, this->channel_axis_);

  if (need_adaptor_) {
    adaptor_ = std::make_shared<ChannelFirstAdaptor>();
    adaptor_->setup(inputs[0], &pre_adaptor_, &post_adaptor_, outputs[0],
                    inputs[0]->shape(), this->batch_axis_, this->channel_axis_,
                    this->ctx_);

    const auto c_ = this->batch_axis_.size();
    batch_size_ = pre_adaptor_.size() / pre_adaptor_.size(c_);
    reduce_size_ =
        pre_adaptor_.size(c_ + 1) * (channel_size_ / this->num_groups_);
    outer_size_ = pre_adaptor_.size() / reduce_size_;
  } else {
    batch_size_ = x->size() / x->size(c);
    reduce_size_ = x->size(c + 1) * (channel_size_ / this->num_groups_);
    outer_size_ = x->size() / reduce_size_;
  }

  a_.reshape({batch_size_ * channel_size_}, true);
  b_.reshape({batch_size_ * channel_size_}, true);
  var_.reshape({batch_size_ * channel_size_}, true);
  mean_.reshape({batch_size_ * channel_size_}, true);

  sum_dy_.reshape({batch_size_ * channel_size_}, true);
  sum_dyx_.reshape({batch_size_ * channel_size_}, true);
  gamma_invstd_.reshape({batch_size_ * channel_size_}, true);
  factor1_.reshape({batch_size_ * this->num_groups_}, true);
  factor2_.reshape({batch_size_ * this->num_groups_}, true);
}

constexpr int GROUP_NORM_ELEMENTWISE_UNROLL_SIZE = 4;

template <typename T, typename index_t>
__global__ void group_norm_forward_normalization_factor(
    const index_t batch_size, const index_t channel_size, const int num_groups,
    const T *mean, const T *var, const T *beta, const T *gamma, T *a, T *b,
    const float eps) {
  // Calculate `a` and `b` of simplified normalization formula
  // as `y = a * x + b`.
  // Original formula is `y = gamma * (x - mean) / sqrt(var) + beta`.
  // Thus `a = gamma / sqrt(var), b = beta - gamma * mean / sqrt(var).`

  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * channel_size) {
    const index_t stats_idx = idx / (channel_size / num_groups);
    const index_t param_idx = idx % channel_size;
    const T scale = gamma ? gamma[param_idx] : (T)1.0f;
    const T bias = beta ? beta[param_idx] : (T)0.0f;

    const T invstd = rsqrt(var[stats_idx] + eps);
    const T scale_invstd = scale * invstd;
    a[idx] = scale_invstd;
    b[idx] = bias - mean[stats_idx] * scale_invstd;
  }
}

template <typename T, typename index_t>
__global__ void
group_norm_forward_normalization(const index_t size, const index_t spatial_size,
                                 const T *x, const T *a, const T *b, T *y) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t ab_idx = idx / spatial_size;
      y[idx] = a[ab_idx] * x[idx] + b[ab_idx];
    }
  }
}

template <typename T, typename index_t>
__global__ void group_norm_backward_gamma_invstd(
    const index_t size, const index_t channel_size, const int num_groups,
    const T *gamma, const T *var, T *gamma_invstd, const float eps) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t stats_idx = idx / (channel_size / num_groups);
      const index_t param_idx = idx % channel_size;
      const T scale = gamma ? gamma[param_idx] : (T)1.0f;
      gamma_invstd[idx] = scale * rsqrt(var[stats_idx] + eps);
    }
  }
}

template <typename T, typename index_t>
__global__ void group_norm_backward_dx_factor(
    const index_t channel_size, const index_t spatial_size,
    const int num_groups, const T *mean, const T *var, const T *dmean,
    const T *dvar, const T *gamma, const T *sum_dy, const T *sum_dyx,
    T *factor1, T *factor2, const float eps) {
  const index_t bidx = blockIdx.x;
  const index_t bidy = blockIdx.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  const index_t chunk_size = channel_size / num_groups;

  // Load and reduce
  float sum_dy_gamma = 0.0f;
  float sum_dyx_gamma = 0.0f;
  for (index_t i = tidx; i < chunk_size; i += bdimx) {
    const index_t idx = (bidx * num_groups + bidy) * chunk_size + i;
    const index_t param_idx = bidy * chunk_size + i;
    const T scale = gamma ? gamma[param_idx] : (T)1.0f;
    sum_dy_gamma += static_cast<float>(sum_dy[idx] * scale);
    sum_dyx_gamma += static_cast<float>(sum_dyx[idx] * scale);
  }

  if (bdimx <= CUDA_WARP_SIZE) {
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dyx_gamma = warpReduceSum(sum_dyx_gamma);
  } else {
    sum_dy_gamma = blockReduceSum(sum_dy_gamma);
    sum_dyx_gamma = blockReduceSum(sum_dyx_gamma);
  }

  // Store
  if (threadIdx.x == 0) {
    const float inv_reduce_size = 1.0f / (chunk_size * spatial_size);
    const index_t stats_idx = bidx * num_groups + bidy;
    const float invstd = rsqrt(var[stats_idx] + eps);
    // TODO:
    const float tmp = (sum_dy_gamma * mean[stats_idx] - sum_dyx_gamma) *
                          invstd * invstd * invstd * inv_reduce_size +
                      (dvar ? 2.0f * dvar[stats_idx] * inv_reduce_size : 0.0f);
    factor1[stats_idx] = tmp;
    factor2[stats_idx] = -tmp * mean[stats_idx] -
                         sum_dy_gamma * invstd * inv_reduce_size +
                         (dmean ? dmean[stats_idx] * inv_reduce_size : 0.0f);
  }
}

template <bool accum, typename T, typename index_t>
__global__ void
group_norm_backward_dx(const index_t size, const index_t channel_size,
                       const index_t spatial_size, const int num_groups,
                       const T *x, const T *dy, const T *gamma_invstd,
                       const T *factor1, const T *factor2, T *dx) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t factor_idx =
          idx / (spatial_size * (channel_size / num_groups));
      const index_t param_idx = idx / (spatial_size);
      if (accum) {
        dx[idx] += gamma_invstd[param_idx] * dy[idx] +
                   factor1[factor_idx] * x[idx] + factor2[factor_idx];
      } else {
        dx[idx] = gamma_invstd[param_idx] * dy[idx] +
                  factor1[factor_idx] * x[idx] + factor2[factor_idx];
      }
    }
  }
}

template <bool beta_accum, bool gamma_accum, typename T, typename index_t>
__global__ void group_norm_backward_dbeta_dgamma(
    const index_t batch_size, const index_t channel_size, const int num_groups,
    const T *mean, const T *var, const T *sum_dy, const T *sum_dyx, T *dbeta,
    T *dgamma, const float eps) {
  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < channel_size) {
    const index_t chunk_size = channel_size / num_groups;

    float db = 0.0f;
    float dg = 0.0f;

    for (index_t n = 0; n < batch_size; n++) {
      const index_t param_idx = n * channel_size + idx;
      const index_t stats_idx = n * num_groups + idx / chunk_size;
      db += static_cast<float>(sum_dy[param_idx]);
      dg += (sum_dyx[param_idx] - sum_dy[param_idx] * mean[stats_idx]) *
            rsqrt(var[stats_idx] + eps);
    }

    if (dbeta) {
      if (beta_accum) {
        dbeta[idx] += db;
      } else {
        dbeta[idx] = db;
      }
    }
    if (dgamma) {
      if (gamma_accum) {
        dgamma[idx] += dg;
      } else {
        dgamma[idx] = dg;
      }
    }
  }
}

template <typename T>
void GroupNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  if (need_adaptor_) {
    // Transpose input to [B, C, H, W] memory format.
    adaptor_->forward_pre(inputs[0], &pre_adaptor_);

    auto in_cf_in = inputs;
    auto in_cf_out = outputs;
    in_cf_in[0] = &pre_adaptor_;
    in_cf_out[0] = &post_adaptor_;

    // Instance normalization
    forward_channel_first(in_cf_in, in_cf_out);

    // Transpose output to original memory format.
    adaptor_->forward_post(&post_adaptor_, outputs[0]);
  } else {
    forward_channel_first(inputs, outputs);
  }
}

template <typename T>
void GroupNormalizationCuda<T>::forward_channel_first(
    const Variables &inputs, const Variables &outputs) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate mean and variance.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_);
    const int num_threads = reduce_size_ < 512 ? 32 : 512; // TODO:

    const auto grid = outer_size_;
    const auto block = num_threads;

    WelfordOp<Tc, Size_t> op(x, mean, var, reduce_size_);
    reduce_2d_x<<<grid, block>>>(op, reduce_size_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate `a` and `b` for simplification of normalization formula
  // as `y = a * x + b`.
  {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *beta = this->no_bias_
                         ? nullptr
                         : inputs[beta_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    Tc *a = a_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *b = b_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid =
        NBLA_CEIL_SIZE_T_DIV(batch_size_ * channel_size_, 256); // TODO:
    const auto block = 256;
    group_norm_forward_normalization_factor<<<grid, block>>>(
        batch_size_, channel_size_, this->num_groups_, mean, var, beta, gamma,
        a, b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Normalization by `y = a * x + b`.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *a = a_.get_data_pointer<Tc>(this->ctx_);
    const Tc *b = b_.get_data_pointer<Tc>(this->ctx_);
    Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const Size_t num_threads = CUDA_WARP_SIZE * 2;

    const auto block = num_threads;
    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, num_threads * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);

    group_norm_forward_normalization<<<grid, block>>>(size, spatial_size, x, a,
                                                      b, y);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

template <typename T>
void GroupNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);

  if (need_adaptor_) {
    adaptor_->backward_post(&post_adaptor_, outputs[0], true, false);

    auto in_cf_in = inputs;
    auto in_cf_out = outputs;
    in_cf_in[0] = &pre_adaptor_;
    in_cf_out[0] = &post_adaptor_;

    auto in_cf_accum = accum;
    in_cf_accum[0] = false;
    backward_channel_first(in_cf_in, in_cf_out, propagate_down, in_cf_accum);

    adaptor_->backward_pre(inputs[0], &pre_adaptor_, propagate_down[0],
                           accum[0]);
  } else {
    backward_channel_first(inputs, outputs, propagate_down, accum);
  }
}

template <typename T>
void GroupNormalizationCuda<T>::backward_channel_first(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate sum of dy and dy*x for the following gradient calculation.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dy = sum_dy_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyx = sum_dyx_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const auto num_threads = spatial_size < 512 ? CUDA_WARP_SIZE : 512; // TODO:

    const auto grid = batch_size_ * channel_size_;
    const auto block = num_threads;

    GNGradOp<Tc, Size_t> op(x, dy, sum_dy, sum_dyx);
    reduce_2d_x<<<grid, block>>>(op, spatial_size);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate gamma / sqrt(var)
  {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    Tc *gamma_invstd = gamma_invstd_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = batch_size_ * channel_size_;
    const auto num_threads = CUDA_WARP_SIZE * 2;

    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, GROUP_NORM_ELEMENTWISE_UNROLL_SIZE * num_threads);
    const auto block = num_threads;

    group_norm_backward_gamma_invstd<<<grid, block>>>(
        size, channel_size_, this->num_groups_, gamma, var, gamma_invstd,
        this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate factor1 and factor2
  {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *dmean = outputs.size() == 3
                          ? v_mean->get_grad_pointer<Tc>(this->ctx_)
                          : nullptr;
    const Tc *dvar =
        outputs.size() == 3 ? v_var->get_grad_pointer<Tc>(this->ctx_) : nullptr;
    const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
    Tc *factor1 = factor1_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *factor2 = factor2_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const auto num_threads = CUDA_WARP_SIZE * 2;

    dim3 grid(batch_size_, this->num_groups_);
    dim3 block(num_threads);

    group_norm_backward_dx_factor<<<grid, block>>>(
        channel_size_, spatial_size, this->num_groups_, mean, var, dmean, dvar,
        gamma, sum_dy, sum_dyx, factor1, factor2, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx by `dx = gamma_invstd * dy + factor1 * x + factor2`.
  if (propagate_down[0]) {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *gamma_invstd = gamma_invstd_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor1 = factor1_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor2 = factor2_.get_data_pointer<Tc>(this->ctx_);
    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const Size_t num_threads = CUDA_WARP_SIZE * 2;

    const auto block = num_threads;
    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, num_threads * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);

    if (accum[0]) {
      group_norm_backward_dx<true><<<grid, block>>>(
          size, channel_size_, spatial_size, this->num_groups_, x, dy,
          gamma_invstd, factor1, factor2, dx);
      NBLA_CUDA_KERNEL_CHECK();
    } else {
      group_norm_backward_dx<false><<<grid, block>>>(
          size, channel_size_, spatial_size, this->num_groups_, x, dy,
          gamma_invstd, factor1, factor2, dx);
      NBLA_CUDA_KERNEL_CHECK();
    }
  }

  if ((inputs.size() > 1 && propagate_down[1]) ||
      (inputs.size() > 2 && propagate_down[2])) {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
    Tc *dbeta = !this->no_bias_ && propagate_down[beta_idx]
                    ? inputs[beta_idx]->cast_grad_and_get_pointer<Tc>(
                          this->ctx_, !accum[beta_idx])
                    : nullptr;
    Tc *dgamma = !this->no_scale_ && propagate_down[gamma_idx]
                     ? inputs[gamma_idx]->cast_grad_and_get_pointer<Tc>(
                           this->ctx_, !accum[gamma_idx])
                     : nullptr;

    const auto block = 256; // TODO:
    const auto grid = NBLA_CEIL_SIZE_T_DIV(channel_size_, 256);

    if (!this->no_bias_ && accum[beta_idx]) {
      if (!this->no_scale_ && accum[gamma_idx]) {
        group_norm_backward_dbeta_dgamma<true, true><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      } else {
        group_norm_backward_dbeta_dgamma<true, false><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      }
    } else {
      if (!this->no_scale_ && accum[gamma_idx]) {
        group_norm_backward_dbeta_dgamma<false, true><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      } else {
        group_norm_backward_dbeta_dgamma<false, false><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      }
    }
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}
