// Copyright 2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/instance_normalization.hpp>
#include <nbla/variable.hpp>

// Kernels and ops
#include <nbla/cuda/function/kernel/instance_normalization.cuh>
#include <nbla/cuda/function/kernel/normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/instance_normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/sum.cuh>
#include <nbla/cuda/utils/reduce_ops/welford.cuh>

namespace nbla {

template <typename T>
void InstanceNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                              const Variables &outputs) {
  InstanceNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto x_shape = x->shape();
  channel_size_ = x_shape[this->channel_axis_];

  // Setup input and output adaptor for channel-last memory format
  need_adaptor_ = ChannelFirstAdaptor::need_adaptor(x_shape, this->batch_axis_,
                                                    this->channel_axis_);

  if (need_adaptor_) {
    adaptor_ = std::make_shared<ChannelFirstAdaptor>();
    adaptor_->setup(x, &pre_adaptor_, &post_adaptor_, outputs[0], x_shape,
                    this->batch_axis_, this->channel_axis_, this->ctx_);

    reduce_size_ = pre_adaptor_.size(this->batch_axis_.size() + 1);
    inv_reduce_size_ = 1.0f / reduce_size_;
    outer_size_ = pre_adaptor_.size() / reduce_size_;
  } else {
    reduce_size_ = inputs[0]->size(this->channel_axis_ + 1);
    inv_reduce_size_ = 1.0f / reduce_size_;
    outer_size_ = inputs[0]->size() / reduce_size_;
  }

  //----------------
  // Reshape buffers
  //----------------

  // Batch stats
  mean_.reshape(Shape_t{outer_size_}, true);
  var_.reshape(Shape_t{outer_size_}, true);

  // Internal buffers for backward calculation
  sum_dy_.reshape(Shape_t{outer_size_}, true);
  sum_dyx_.reshape(Shape_t{outer_size_}, true);
  factor_a_.reshape(Shape_t{outer_size_}, true);
  factor_b_.reshape(Shape_t{outer_size_}, true);
}

template <typename T>
void InstanceNormalizationCuda<T>::forward_impl(const Variables &inputs,
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

    // Instance normalization
    forward_channel_first(channel_first_inputs, channel_first_outputs);

    // Transpose output to original memory format.
    adaptor_->convert_from_channel_first(&post_adaptor_, outputs[0]);
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
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const int num_threads = reduce_size_ < NBLA_CUDA_IN_NUM_THREADS
                                ? CUDA_WARP_SIZE
                                : NBLA_CUDA_IN_NUM_THREADS;

    const auto grid =
        std::min(outer_size_, static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));
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

    const size_t elements_per_grid_y = NBLA_CUDA_IN_NUM_THREADS * 4;
    dim3 grid;
    grid.x =
        std::min(outer_size_, static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = NBLA_CUDA_IN_NUM_THREADS;

    auto kernel =
        instance_norm_forward_normalization<Tc, Size_t, false /* bc_beta */,
                                            false /* bc_gamma */>;
    if (this->need_beta_broadcast_) {
      kernel =
          this->need_gamma_broadcast_
              ? instance_norm_forward_normalization<Tc, Size_t, true, true>
              : instance_norm_forward_normalization<Tc, Size_t, true, false>;
    } else {
      kernel =
          this->need_gamma_broadcast_
              ? instance_norm_forward_normalization<Tc, Size_t, false, true>
              : instance_norm_forward_normalization<Tc, Size_t, false, false>;
    }
    kernel<<<grid, block>>>(outer_size_, channel_size_, reduce_size_, x, mean,
                            var, beta, gamma, y, this->eps_);
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
void InstanceNormalizationCuda<T>::backward_channel_first(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  // Calculate sum of dy and sum of dy * x.
  // These values are used in dx and dbeta/dgamma calculation.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dy = sum_dy_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyx = sum_dyx_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const int num_threads = reduce_size_ < NBLA_CUDA_IN_NUM_THREADS
                                ? CUDA_WARP_SIZE
                                : NBLA_CUDA_IN_NUM_THREADS;

    const auto grid =
        std::min(outer_size_, static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));

    const auto block = num_threads;

    INGradOp<Tc, Size_t> op(x, dy, sum_dy, sum_dyx);
    reduce_2d_x<<<grid, block>>>(op, outer_size_, reduce_size_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx
  if (propagate_down[0]) {
    backward_channel_first_dx(inputs, outputs, propagate_down, accum);
  }

  // Calculate dbeta and dgamma.
  if ((inputs.size() > 1 && propagate_down[1]) ||
      (inputs.size() > 2 && propagate_down[2])) {
    backward_channel_first_dbeta_dgamma(inputs, outputs, propagate_down, accum);
  }

  // Clear internal buffer
  sum_dy_.data()->array()->clear();
  sum_dyx_.data()->array()->clear();
}

template <typename T>
void InstanceNormalizationCuda<T>::backward_channel_first_dx(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

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

    Tc *factor_a = factor_a_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *factor_b = factor_b_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const auto grid = std::min(static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS),
                               static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(
                                   outer_size_, NBLA_CUDA_IN_NUM_THREADS)));
    const auto block = NBLA_CUDA_IN_NUM_THREADS;

    auto kernel = this->need_gamma_broadcast_
                      ? instance_norm_backward_dx_factor<Tc, Size_t, true>
                      : instance_norm_backward_dx_factor<Tc, Size_t, false>;
    kernel<<<grid, block>>>(outer_size_, channel_size_, inv_reduce_size_, gamma,
                            mean, var, dmean, dvar, sum_dy, sum_dyx, factor_a,
                            factor_b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx.
  {
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

    const size_t elements_per_grid_y = NBLA_CUDA_IN_NUM_THREADS * 4;
    dim3 grid;
    grid.x =
        std::min(outer_size_, static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = NBLA_CUDA_IN_NUM_THREADS;

    auto kernel = instance_norm_backward_dx<Tc, Size_t, false /* bc_gamma */,
                                            false /* accum */>;
    if (this->need_gamma_broadcast_) {
      kernel = accum[0] ? instance_norm_backward_dx<Tc, Size_t, true, true>
                        : instance_norm_backward_dx<Tc, Size_t, true, false>;
    } else {
      kernel = accum[0] ? instance_norm_backward_dx<Tc, Size_t, false, true>
                        : instance_norm_backward_dx<Tc, Size_t, false, false>;
    }
    kernel<<<grid, block>>>(outer_size_, channel_size_, reduce_size_, x, gamma,
                            dy, var, factor_a, factor_b, dx, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();

    // Clear internal buffer
    factor_a_.data()->array()->clear();
    factor_b_.data()->array()->clear();
  }
}

template <typename T>
void InstanceNormalizationCuda<T>::backward_channel_first_dbeta_dgamma(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate dbeta and dgamma.
  Variable dbeta_buf(Shape_t{outer_size_});
  Variable dgamma_buf(Shape_t{outer_size_});

  const auto beta_idx = 1;
  const auto gamma_idx = this->no_bias_ ? 1 : 2;

  const bool pd_beta = !this->no_bias_ && propagate_down[beta_idx];
  const bool pd_gamma = !this->no_scale_ && propagate_down[gamma_idx];
  const bool accum_beta = !this->no_bias_ && accum[beta_idx];
  const bool accum_gamma = !this->no_scale_ && accum[gamma_idx];
  const bool bc_beta = this->need_beta_broadcast_;
  const bool bc_gamma = this->need_gamma_broadcast_;

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *gamma = this->no_scale_
                        ? nullptr
                        : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
  const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
  const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
  const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);

  // Support function to get pointer of affine parameter grad.
  auto cast_affine_param_grad = [&](Variable &buffer, const int inputs_idx,
                                    const bool propagate_down, const bool accum,
                                    const bool need_broadcast) {
    if (!propagate_down) {
      return (Tc *)nullptr;
    }

    if (need_broadcast) {
      return buffer.cast_grad_and_get_pointer<Tc>(this->ctx_, true);
    }

    return inputs[inputs_idx]->cast_grad_and_get_pointer<Tc>(this->ctx_,
                                                             !accum);
  };

  Tc *dbeta =
      cast_affine_param_grad(dbeta_buf, beta_idx, pd_beta, accum_beta, bc_beta);
  Tc *dgamma = cast_affine_param_grad(dgamma_buf, gamma_idx, pd_gamma,
                                      accum_gamma, bc_gamma);

  const auto grid = std::min(static_cast<Size_t>(NBLA_CUDA_IN_MAX_BLOCKS),
                             static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(
                                 outer_size_, NBLA_CUDA_IN_NUM_THREADS)));
  const auto block = NBLA_CUDA_IN_NUM_THREADS;

  // Select kernels by accum combination.
  // If broadcast is needed, accumulation is not done here but done after
  // broadcast backward.
  auto kernel =
      instance_norm_backward_dbeta_dgamma<false /* accum_beta */,
                                          false /* accum_gamma */, Tc, Size_t>;
  if (accum_beta && !bc_beta) {
    kernel = accum_gamma && !bc_gamma
                 ? instance_norm_backward_dbeta_dgamma<true, true, Tc, Size_t>
                 : instance_norm_backward_dbeta_dgamma<true, false, Tc, Size_t>;
  } else {
    kernel =
        accum_gamma && !bc_gamma
            ? instance_norm_backward_dbeta_dgamma<false, true, Tc, Size_t>
            : instance_norm_backward_dbeta_dgamma<false, false, Tc, Size_t>;
  }
  kernel<<<grid, block>>>(outer_size_, reduce_size_, x, gamma, dy, sum_dy,
                          sum_dyx, mean, var, dbeta, dgamma, this->eps_);
  NBLA_CUDA_KERNEL_CHECK();

  // Support function for affine param broadcast backward.
  auto affine_param_broadcast_backward = [&](
      Tc *d_affine_param_ptr, const int param_idx, const bool accum) {
    // Reduction as broadcast backward
    ReduceSetup reduce_setup;
    reduce_setup(Shape_t{outer_size_ / channel_size_, channel_size_},
                 Shape_t{0});

    // d_beta or d_gamma
    auto *d_affine_param_out_ptr =
        inputs[param_idx]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum);

    if (accum) {
      // Temporary buffer for reduction
      Variable reduce_buf(Shape_t{channel_size_});
      auto *reduce_buf_ptr =
          reduce_buf.cast_grad_and_get_pointer<Tc>(this->ctx_, true);
      device_sum(this->ctx_, d_affine_param_ptr, reduce_buf_ptr, reduce_setup);

      // Gradient accumulation
      auto kernel =
          accum ? instance_norm_backward_accum_affine_param<Tc, Size_t, true>
                : instance_norm_backward_accum_affine_param<Tc, Size_t, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, channel_size_, reduce_buf_ptr,
                                     d_affine_param_out_ptr);
    } else {
      device_sum(this->ctx_, d_affine_param_ptr, d_affine_param_out_ptr,
                 reduce_setup);
    }
  };

  if (pd_beta && bc_beta) {
    affine_param_broadcast_backward(dbeta, beta_idx, accum_beta);
  }

  if (pd_gamma && bc_gamma) {
    affine_param_broadcast_backward(dgamma, gamma_idx, accum_gamma);
  }
}
}
