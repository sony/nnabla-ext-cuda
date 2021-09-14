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
#include <nbla/cuda/function/instance_normalization.hpp>
#include <nbla/variable.hpp>

// Common kernels and reduce ops
#include <nbla/cuda/function/kernel/normalization.cuh>

namespace nbla {

template <typename T>
void InstanceNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                              const Variables &outputs) {
  InstanceNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  need_adaptor_ = ChannelFirstAdaptor::need_adaptor(
      inputs[0]->shape(), this->batch_axis_, this->channel_axis_);

  if (need_adaptor_) {
    adaptor_ = std::make_shared<ChannelFirstAdaptor>();
    adaptor_->setup(inputs[0], &pre_adaptor_, &post_adaptor_, outputs[0],
                    inputs[0]->shape(), this->batch_axis_, this->channel_axis_,
                    this->ctx_);

    reduce_size_ = pre_adaptor_.size(this->batch_axis_.size() + 1);
    outer_size_ = pre_adaptor_.size() / reduce_size_;
  } else {
    reduce_size_ = inputs[0]->size(this->channel_axis_ + 1);
    outer_size_ = inputs[0]->size() / reduce_size_;
  }

  mean_.reshape({outer_size_}, true);
  var_.reshape({outer_size_}, true);
  sum_dy_.reshape({outer_size_}, true);
  sum_dyx_.reshape({outer_size_}, true);
  factor_a_.reshape({outer_size_}, true);
  factor_b_.reshape({outer_size_}, true);
}

template <typename T, typename index_t>
__global__ void
instance_norm_forward_normalization(const index_t outer_size,
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
      const T scale = gamma ? gamma[outer_idx] : (T)1.0f;
      const T bias = beta ? beta[outer_idx] : (T)0.0f;
      const T invstd = rsqrt(var[outer_idx] + eps);

      y[idx] = scale * invstd * (x[idx] - mean[outer_idx]) + bias;
    }
  }
}

template <typename T, typename index_t>
__global__ void instance_norm_backward_dx_factor(
    const index_t outer_size, const index_t reduce_size, const T *gamma,
    const T *mean, const T *var, const T *dmean, const T *dvar, const T *sum_dy,
    const T *sum_dyx, T *factor_a, T *factor_b, const float eps) {
  // Grid-stride loop
  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outer_size;
       idx += gridDim.x * blockDim.x) {
    const float inv_reduce_size = 1.0f / reduce_size;
    const float invstd = rsqrt(var[idx] + eps);
    const float scale = gamma ? static_cast<float>(gamma[idx]) : 1.0f;

    const float tmp = (sum_dy[idx] * scale * mean[idx] - sum_dyx[idx] * scale) *
                          invstd * invstd * invstd * inv_reduce_size +
                      (dvar ? 2.0f * dvar[idx] * inv_reduce_size : 0.0f);

    factor_a[idx] = tmp;
    factor_b[idx] = -tmp * mean[idx] -
                    sum_dy[idx] * scale * invstd * inv_reduce_size +
                    (dmean ? dmean[idx] * inv_reduce_size : 0.0f);
  }
}

template <bool accum, typename T, typename index_t>
__global__ void
instance_norm_backward_dx(const index_t outer_size, const index_t reduce_size,
                          const T *x, const T *gamma, const T *dy, const T *var,
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
      const T scale = gamma ? gamma[outer_idx] : (T)1.0f;
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
__global__ void instance_norm_backward_dbeta_dgamma(
    const index_t outer_size, const index_t reduce_size, const T *x,
    const T *gamma, const T *dy, const T *sum_dy, const T *sum_dyx,
    const T *mean, const T *var, T *dbeta_out, T *dgamma_out, const float eps) {
  // Grid-stride loop
  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outer_size;
       idx += gridDim.x * blockDim.x) {
    const float invstd = rsqrt(var[idx] + eps);
    const float dbeta = sum_dy[idx];
    const float dgamma =
        sum_dyx[idx] * invstd - sum_dy[idx] * mean[idx] * invstd;

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
void InstanceNormalizationCuda<T>::forward_impl(const Variables &inputs,
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
void InstanceNormalizationCuda<T>::forward_channel_first(
    const Variables &inputs, const Variables &outputs) {
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

    const int num_threads =
        reduce_size_ < IN_NUM_THREADS ? CUDA_WARP_SIZE : IN_NUM_THREADS;

    const auto grid = std::min(outer_size_, static_cast<Size_t>(IN_MAX_BLOCKS));
    const auto block = num_threads;

    WelfordOp<Tc, Size_t> op(x, mean, var, reduce_size_);
    reduce_2d_x<<<grid, block>>>(op, outer_size_, reduce_size_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Instance normalization
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

    const size_t elements_per_grid_y = IN_NUM_THREADS * 4;
    dim3 grid;
    grid.x = std::min(outer_size_, static_cast<Size_t>(IN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(IN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = IN_NUM_THREADS;

    instance_norm_forward_normalization<<<grid, block>>>(
        outer_size_, reduce_size_, x, mean, var, beta, gamma, y, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

template <typename T>
void InstanceNormalizationCuda<T>::backward_impl(
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
void InstanceNormalizationCuda<T>::backward_channel_first(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate sum of dy and sum of dy * x.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dy = sum_dy_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyx = sum_dyx_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const int num_threads =
        reduce_size_ < IN_NUM_THREADS ? CUDA_WARP_SIZE : IN_NUM_THREADS;

    const auto grid = std::min(outer_size_, static_cast<Size_t>(IN_MAX_BLOCKS));

    const auto block = num_threads;

    INGradOp<Tc, Size_t> op(x, dy, sum_dy, sum_dyx);
    reduce_2d_x<<<grid, block>>>(op, outer_size_, reduce_size_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // TODO: change the comment
  // Calculate a and b such that `dx = gamma / sqrt(var) * dy + a * x + b`.
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

    Tc *factor_a = factor_a_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *factor_b = factor_b_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid = std::min(
        static_cast<Size_t>(IN_MAX_BLOCKS),
        static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(outer_size_, IN_NUM_THREADS)));
    const auto block = IN_NUM_THREADS;

    instance_norm_backward_dx_factor<<<grid, block>>>(
        outer_size_, reduce_size_, gamma, mean, var, dmean, dvar, sum_dy,
        sum_dyx, factor_a, factor_b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx.
  if (propagate_down[0]) {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *factor_a = factor_a_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor_b = factor_b_.get_data_pointer<Tc>(this->ctx_);

    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);

    const size_t elements_per_grid_y = IN_NUM_THREADS * 4;
    dim3 grid;
    grid.x = std::min(outer_size_, static_cast<Size_t>(IN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(IN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = IN_NUM_THREADS;

    if (accum[0]) {
      instance_norm_backward_dx<true><<<grid, block>>>(
          outer_size_, reduce_size_, x, gamma, dy, var, factor_a, factor_b, dx,
          this->eps_);
    } else {
      instance_norm_backward_dx<false><<<grid, block>>>(
          outer_size_, reduce_size_, x, gamma, dy, var, factor_a, factor_b, dx,
          this->eps_);
    }
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dbeta and dgamma.
  if ((inputs.size() > 1 && propagate_down[1]) ||
      (inputs.size() > 2 && propagate_down[2])) {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
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

    const auto grid = std::min(
        static_cast<Size_t>(IN_MAX_BLOCKS),
        static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(outer_size_, IN_NUM_THREADS)));
    const auto block = IN_NUM_THREADS;

    if (!this->no_bias_ && accum[beta_idx]) {
      if (!this->no_scale_ && accum[gamma_idx]) {
        instance_norm_backward_dbeta_dgamma<true, true><<<grid, block>>>(
            outer_size_, reduce_size_, x, gamma, dy, sum_dy, sum_dyx, mean, var,
            dbeta, dgamma, this->eps_);
      } else {
        instance_norm_backward_dbeta_dgamma<true, false><<<grid, block>>>(
            outer_size_, reduce_size_, x, gamma, dy, sum_dy, sum_dyx, mean, var,
            dbeta, dgamma, this->eps_);
      }
    } else {
      if (!this->no_scale_ && accum[gamma_idx]) {
        instance_norm_backward_dbeta_dgamma<false, true><<<grid, block>>>(
            outer_size_, reduce_size_, x, gamma, dy, sum_dy, sum_dyx, mean, var,
            dbeta, dgamma, this->eps_);
      } else {
        instance_norm_backward_dbeta_dgamma<false, false><<<grid, block>>>(
            outer_size_, reduce_size_, x, gamma, dy, sum_dy, sum_dyx, mean, var,
            dbeta, dgamma, this->eps_);
      }
    }
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}
