// Copyright 2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

// Kernels and ops
#include <nbla/cuda/function/kernel/layer_normalization.cuh>
#include <nbla/cuda/function/kernel/normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/layer_normalization.cuh>
#include <nbla/cuda/utils/reduce_ops/welford.cuh>

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

  // Check if `batch_axis` are continuously aligned in outer axes.
  // Currently, `batch_axis` not satisfy above condition like `[0, 2]` or `[1,
  // 2]` are not supported in CUDA backend.
  const auto &ba = this->batch_axis_;
  bool need_fall_back = *std::max_element(ba.begin(), ba.end()) >= ba.size();
  if (need_fall_back) {
    this->fall_back_func_ = make_shared<LayerNormalization<T>>(
        this->ctx_, this->batch_axis_, this->eps_, this->no_scale_,
        this->no_bias_);
    this->fall_back_func_->setup(inputs, outputs);
    return;
  }

  batch_size_ = 1;
  for (const auto b : this->batch_axis_) {
    batch_size_ *= x_shape[b];
  }

  reduce_size_ = x_size / batch_size_;
  inv_reduce_size_ = 1.0f / reduce_size_;

  //----------------
  // Reshape buffers
  //----------------

  // Batch stats
  mean_.reshape(Shape_t{batch_size_}, true);
  var_.reshape(Shape_t{batch_size_}, true);

  // Internal buffers for backward calculation
  sum_dygamma_.reshape(Shape_t{batch_size_}, true);
  sum_dyxgamma_.reshape(Shape_t{batch_size_}, true);
  factor_a_.reshape(Shape_t{batch_size_}, true);
  factor_b_.reshape(Shape_t{batch_size_}, true);
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
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const auto grid =
        std::min(batch_size_, static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

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
    Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const size_t elements_per_grid_y = NBLA_CUDA_LN_NUM_THREADS * 4;
    dim3 grid;
    grid.x =
        std::min(batch_size_, static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

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
    Tc *sum_dygamma =
        sum_dygamma_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *sum_dyxgamma =
        sum_dyxgamma_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const auto grid =
        std::min(batch_size_, static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

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

    Tc *factor_a = factor_a_.cast_data_and_get_pointer<Tc>(this->ctx_, true);
    Tc *factor_b = factor_b_.cast_data_and_get_pointer<Tc>(this->ctx_, true);

    const auto grid = std::min(static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS),
                               static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(
                                   batch_size_, NBLA_CUDA_LN_NUM_THREADS)));
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

    layer_norm_backward_dx_factor<<<grid, block>>>(
        batch_size_, inv_reduce_size_, mean, var, dmean, dvar, sum_dygamma,
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

    const Size_t elements_per_grid_y = NBLA_CUDA_LN_NUM_THREADS * 4;
    dim3 grid;
    grid.x =
        std::min(batch_size_, static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    grid.y = std::min(NBLA_CEIL_SIZE_T_DIV(reduce_size_, elements_per_grid_y),
                      static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS));
    grid.z = 1;
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

    auto kernel = accum[0] ? layer_norm_backward_dx<true, Tc, Size_t>
                           : layer_norm_backward_dx<false, Tc, Size_t>;
    kernel<<<grid, block>>>(batch_size_, reduce_size_, x, dy, gamma, var,
                            factor_a, factor_b, dx, this->eps_);
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

    const auto grid = std::min(static_cast<Size_t>(NBLA_CUDA_LN_MAX_BLOCKS),
                               static_cast<Size_t>(NBLA_CEIL_SIZE_T_DIV(
                                   reduce_size_, NBLA_CUDA_LN_NUM_THREADS)));
    const auto block = NBLA_CUDA_LN_NUM_THREADS;

    // Select kernels by accum combination.
    auto kernel = layer_norm_backward_dbeta_dgamma<true, true, Tc, Size_t>;
    if (!this->no_bias_ && accum[beta_idx]) {
      kernel = !this->no_scale_ && accum[gamma_idx]
                   ? layer_norm_backward_dbeta_dgamma<true, true, Tc, Size_t>
                   : layer_norm_backward_dbeta_dgamma<true, false, Tc, Size_t>;
    } else {
      kernel = !this->no_scale_ && accum[gamma_idx]
                   ? layer_norm_backward_dbeta_dgamma<false, true, Tc, Size_t>
                   : layer_norm_backward_dbeta_dgamma<false, false, Tc, Size_t>;
    }
    kernel<<<grid, block>>>(batch_size_, reduce_size_, x, dy, mean, var, dbeta,
                            dgamma, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}
