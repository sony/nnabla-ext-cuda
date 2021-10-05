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

// Kernels and ops
#include <nbla/cuda/function/kernel/group_normalization.cuh>
#include <nbla/cuda/function/kernel/normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/group_normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/welford.cuh>

namespace nbla {

template <typename T>
void GroupNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  GroupNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto x_shape = x->shape();
  const auto ndim = x->ndim();

  // Setup input and output adaptor for channel-last memory format
  need_adaptor_ = ChannelFirstAdaptor::need_adaptor(
      inputs[0]->shape(), this->batch_axis_, this->channel_axis_);

  if (need_adaptor_) {
    adaptor_ = std::make_shared<ChannelFirstAdaptor>();
    adaptor_->setup(inputs[0], &pre_adaptor_, &post_adaptor_, outputs[0],
                    inputs[0]->shape(), this->batch_axis_, this->channel_axis_,
                    this->ctx_);

    const auto c = this->batch_axis_.size();
    channel_size_ = pre_adaptor_.shape()[c];
    batch_size_ = pre_adaptor_.size() / pre_adaptor_.size(c);
    reduce_size_ =
        pre_adaptor_.size(c + 1) * (channel_size_ / this->num_groups_);
    inv_reduce_size_ = 1.0f / reduce_size_;
    outer_size_ = pre_adaptor_.size() / reduce_size_;
  } else {
    const auto c = this->channel_axis_;
    channel_size_ = x_shape[c];
    batch_size_ = x->size() / x->size(c);
    reduce_size_ = x->size(c + 1) * (channel_size_ / this->num_groups_);
    inv_reduce_size_ = 1.0f / reduce_size_;
    outer_size_ = x->size() / reduce_size_;
  }

  //----------------
  // Reshape buffers
  //----------------

  // Batch stats
  var_.reshape({batch_size_ * channel_size_}, true);
  mean_.reshape({batch_size_ * channel_size_}, true);

  // Internal buffers for forward calculation
  a_.reshape({batch_size_ * channel_size_}, true);
  b_.reshape({batch_size_ * channel_size_}, true);

  // Internal buffers for backward calculation
  sum_dy_.reshape({batch_size_ * channel_size_}, true);
  sum_dyx_.reshape({batch_size_ * channel_size_}, true);
  gamma_invstd_.reshape({batch_size_ * channel_size_}, true);
  factor1_.reshape({batch_size_ * this->num_groups_}, true);
  factor2_.reshape({batch_size_ * this->num_groups_}, true);
}

template <typename T>
void GroupNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  // Currently, only channel-fist kernels are provided. Channel-last execution
  // is performed by transforming input and output memory format to
  // channel-first and using channel-first implementation. The transformation is
  // performed by ChannelFirstAdaptor.
  if (need_adaptor_) {
    // Transpose input to [B, C, H, W] memory format.
    adaptor_->convert_to_channel_first(inputs[0], &pre_adaptor_);

    auto channel_first_inputs = inputs;
    auto channel_first_outputs = outputs;
    channel_first_inputs[0] = &pre_adaptor_;
    channel_first_outputs[0] = &post_adaptor_;

    // Group normalization
    forward_channel_first(channel_first_inputs, channel_first_outputs);

    // Transpose output to original memory format.
    adaptor_->convert_from_channel_first(&post_adaptor_, outputs[0]);
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
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_, true);
    const int num_threads = reduce_size_ < NBLA_CUDA_GN_NUM_THREADS
                                ? CUDA_WARP_SIZE
                                : NBLA_CUDA_GN_NUM_THREADS;

    const auto grid =
        std::min(outer_size_, static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
    const auto block = num_threads;

    WelfordOp<Tc, Size_t> op(x, mean, var, reduce_size_);
    reduce_2d_x<<<grid, block>>>(op, outer_size_, reduce_size_);
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
    Tc *a = a_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *b = b_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const auto block = NBLA_CUDA_GN_NUM_THREADS;
    const auto grid = std::min(NBLA_CEIL_SIZE_T_DIV(batch_size_ * channel_size_,
                                                    NBLA_CUDA_GN_NUM_THREADS),
                               static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
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
    Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const Size_t num_threads = CUDA_WARP_SIZE * 2;

    const auto block = num_threads;
    const auto grid = std::min(
        NBLA_CEIL_SIZE_T_DIV(size, num_threads * NBLA_CUDA_GN_N_UNROLL),
        static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));

    group_norm_forward_normalization<Tc, Size_t,
                                     NBLA_CUDA_GN_N_UNROLL><<<grid, block>>>(
        size, spatial_size, x, a, b, y);
    NBLA_CUDA_KERNEL_CHECK();

    // Clear internal buffers
    a_.data()->array()->clear();
    b_.data()->array()->clear();
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
    adaptor_->convert_from_channel_first_backward(&post_adaptor_, outputs[0],
                                                  true, false);

    auto channel_first_inputs = inputs;
    auto channel_first_outputs = outputs;
    channel_first_inputs[0] = &pre_adaptor_;
    channel_first_outputs[0] = &post_adaptor_;

    auto channel_first_accum = accum;
    channel_first_accum[0] = false;
    backward_channel_first(channel_first_inputs, channel_first_outputs,
                           propagate_down, channel_first_accum);

    post_adaptor_.data()->array()->clear();
    post_adaptor_.grad()->array()->clear();

    adaptor_->convert_to_channel_first_backward(inputs[0], &pre_adaptor_,
                                                propagate_down[0], accum[0]);

    pre_adaptor_.data()->array()->clear();
    pre_adaptor_.grad()->array()->clear();
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
    Tc *sum_dy = sum_dy_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *sum_dyx = sum_dyx_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const Size_t size = inputs[0]->size();
    const Size_t bc_size = batch_size_ * channel_size_;
    const Size_t spatial_size = size / bc_size;
    const auto num_threads = spatial_size < NBLA_CUDA_GN_NUM_THREADS
                                 ? CUDA_WARP_SIZE
                                 : NBLA_CUDA_GN_NUM_THREADS;

    const auto grid =
        std::min(bc_size, static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
    const auto block = num_threads;

    GNGradOp<Tc, Size_t> op(x, dy, sum_dy, sum_dyx);
    reduce_2d_x<<<grid, block>>>(op, bc_size, spatial_size);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate gamma / sqrt(var)
  if (propagate_down[0]) {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    Tc *gamma_invstd =
        gamma_invstd_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const Size_t size = batch_size_ * channel_size_;
    const auto num_threads = CUDA_WARP_SIZE * 2;

    const auto grid = std::min(
        NBLA_CEIL_SIZE_T_DIV(size, num_threads * NBLA_CUDA_GN_N_UNROLL),
        static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
    const auto block = num_threads;

    group_norm_backward_gamma_invstd<Tc, Size_t,
                                     NBLA_CUDA_GN_N_UNROLL><<<grid, block>>>(
        size, channel_size_, this->num_groups_, gamma, var, gamma_invstd,
        this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate factor1 and factor2
  if (propagate_down[0]) {
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
    Tc *factor1 = factor1_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *factor2 = factor2_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const auto num_threads = CUDA_WARP_SIZE * 2;

    dim3 grid;
    grid.x =
        std::min(batch_size_, static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
    grid.y = std::min(static_cast<Size_t>(this->num_groups_),
                      static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));
    dim3 block(num_threads);

    group_norm_backward_dx_factor<<<grid, block>>>(
        batch_size_, channel_size_, spatial_size, inv_reduce_size_,
        this->num_groups_, mean, var, dmean, dvar, gamma, sum_dy, sum_dyx,
        factor1, factor2, this->eps_);
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
    const auto grid = std::min(
        NBLA_CEIL_SIZE_T_DIV(size, num_threads * NBLA_CUDA_GN_N_UNROLL),
        static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));

    auto kernel =
        accum[0]
            ? group_norm_backward_dx<true, Tc, Size_t, NBLA_CUDA_GN_N_UNROLL>
            : group_norm_backward_dx<false, Tc, Size_t, NBLA_CUDA_GN_N_UNROLL>;
    kernel<<<grid, block>>>(size, channel_size_, spatial_size,
                            this->num_groups_, x, dy, gamma_invstd, factor1,
                            factor2, dx);
    NBLA_CUDA_KERNEL_CHECK();

    // Clear internal buffer
    gamma_invstd_.data()->array()->clear();
    factor1_.data()->array()->clear();
    factor2_.data()->array()->clear();
  }

  // Calculate dbeta and dgamma.
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

    const auto block = NBLA_CUDA_GN_NUM_THREADS;
    const auto grid =
        std::min(NBLA_CEIL_SIZE_T_DIV(channel_size_, NBLA_CUDA_GN_NUM_THREADS),
                 static_cast<Size_t>(NBLA_CUDA_GN_MAX_BLOCKS));

    // Select kernels by accum combination.
    auto kernel = group_norm_backward_dbeta_dgamma<true, true, Tc, Size_t>;
    if (!this->no_bias_ && accum[beta_idx]) {
      kernel = !this->no_scale_ && accum[gamma_idx]
                   ? group_norm_backward_dbeta_dgamma<true, true, Tc, Size_t>
                   : group_norm_backward_dbeta_dgamma<true, false, Tc, Size_t>;
    } else {
      kernel = !this->no_scale_ && accum[gamma_idx]
                   ? group_norm_backward_dbeta_dgamma<false, true, Tc, Size_t>
                   : group_norm_backward_dbeta_dgamma<false, false, Tc, Size_t>;
    }
    kernel<<<grid, block>>>(batch_size_, channel_size_, this->num_groups_, mean,
                            var, sum_dy, sum_dyx, dbeta, dgamma, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Clear internal buffer
  sum_dy_.data()->array()->clear();
  sum_dyx_.data()->array()->clear();
}
}
