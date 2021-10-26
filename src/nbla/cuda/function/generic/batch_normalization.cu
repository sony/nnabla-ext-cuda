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
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

//#include <nbla/cuda/function/kernel/batch_normalization.cuh>
#include "kernel/batch_normalization.cu"

#define BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION

namespace nbla {

template <typename T>
void BatchNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  BatchNormalization<T>::setup_impl(inputs, outputs);
  v_dmean_.reshape(Shape_t{this->size1_}, true);
  v_dvar_.reshape(Shape_t{this->size1_}, true);
#ifdef BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION
  // setup for transpose
  const int ndim = inputs[0]->ndim();

  // for transpose
  v_axes_.reshape(Shape_t{ndim}, true);
  v_in_strides_.reshape(Shape_t{ndim}, true);
  v_out_strides_.reshape(Shape_t{ndim}, true);
  v_in_shape_.reshape(Shape_t{ndim}, true);
  v_out_shape_.reshape(Shape_t{ndim}, true);
  v_din_trans_.reshape(inputs[0]->shape(), true);

  // work memory for data of each axis
  v_inv_sqrt_variance_.reshape(Shape_t{this->size1_}, true);
  v_t_.reshape(Shape_t{this->size1_}, true);

  // work memory for each block data of shuffle reduction
  this->blocks =
      min((this->size02_ + NBLA_CUDA_NUM_THREADS - 1) / NBLA_CUDA_NUM_THREADS,
          Size_t{1024});
  v_mean_reduction_space_.reshape(Shape_t{blocks}, true);
  v_variance_reduction_space_.reshape(Shape_t{blocks}, true);
  v_tmp_reduction_space_.reshape(Shape_t{blocks}, true);

  // make shape for transpose
  Context cpu; // CPU Context
  int *p_axes = v_axes_.cast_data_and_get_pointer<int>(cpu, true);
  int *p_in_strides = v_in_strides_.cast_data_and_get_pointer<int>(cpu, true);
  int *p_out_strides = v_out_strides_.cast_data_and_get_pointer<int>(cpu, true);
  int *p_out_shape = v_out_shape_.cast_data_and_get_pointer<int>(cpu, true);
  int *p_in_shape = v_in_shape_.cast_data_and_get_pointer<int>(cpu, true);
  for (int i = 0; i < ndim; p_axes[i] = i, ++i)
    ;
  if (this->axes_[0] != 0) {
    p_axes[0] = this->axes_[0];
    p_axes[this->axes_[0]] = 0;
  }
  Shape_t shape(ndim);
  for (int i = 0; i < ndim; ++i)
    shape[i] = inputs[0]->shape()[p_axes[i]];
  v_in_trans_.reshape(shape, true);
  for (int i = 0; i < ndim; ++i) {
    p_in_strides[i] = inputs[0]->strides()[i];
    p_out_strides[i] = v_in_trans_.strides()[i];
    p_in_shape[i] = inputs[0]->shape()[i];
    p_out_shape[i] = v_in_trans_.shape()[i];
  }
#endif
}

template <class T>
void BatchNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->batch_stat_) { // Training mode.
    forward_impl_batch(inputs, outputs, true /* update_inputs */);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <class T>
void BatchNormalizationCuda<T>::recompute_impl(const Variables &inputs,
                                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->batch_stat_) { // Training mode.
    forward_impl_batch(inputs, outputs, false /* update_inputs */);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <class T>
void BatchNormalizationCuda<T>::forward_impl_batch(const Variables &inputs,
                                                   const Variables &outputs,
                                                   const bool update_inputs) {
  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }
  // Inputs
  const int b_idx = this->b_idx_;
  const int g_idx = this->g_idx_;
  const int m_idx = this->m_idx_;
  const int v_idx = this->v_idx_;
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *beta = this->no_bias_
                       ? nullptr
                       : inputs[b_idx]->get_data_pointer<Tc>(this->ctx_);
  const Tc *gamma = this->no_scale_
                        ? nullptr
                        : inputs[g_idx]->get_data_pointer<Tc>(this->ctx_);
  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Tc *m =
      batch_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true); // batch mean
  Tc *v =
      batch_var->cast_data_and_get_pointer<Tc>(this->ctx_, true); // batch varf
  // Inputs/Outputs
  Tc *rm = !update_inputs ? nullptr
                          : inputs[m_idx]->cast_data_and_get_pointer<Tc>(
                                this->ctx_); // running mean
  Tc *rv = !update_inputs ? nullptr
                          : inputs[v_idx]->cast_data_and_get_pointer<Tc>(
                                this->ctx_); // running var

#ifdef BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION
  const int ndim = inputs[0]->ndim();
  auto get_ = [this](Variable &var) {
    return var.get_data_pointer<int>(this->ctx_);
  };
  auto get_data_ptr_ = [this](Variable &var) {
    return var.cast_data_and_get_pointer<Tc>(this->ctx_);
  };
  const int *axes = get_(this->v_axes_);
  const int *in_strides = get_(this->v_in_strides_);
  const int *out_strides = get_(this->v_out_strides_);
  const int *in_shape = get_(this->v_in_shape_);
  const int *out_shape = get_(this->v_out_shape_);
  Tc *in_trans = get_data_ptr_(this->v_in_trans_);
  Tc *mean_reduction_space = get_data_ptr_(this->v_mean_reduction_space_);
  Tc *variance_reduction_space =
      get_data_ptr_(this->v_variance_reduction_space_);
  Tc *inv_sqrt_variance = get_data_ptr_(this->v_inv_sqrt_variance_);
  forward_batch_parallel_reduction(
      this->size0_, this->size1_, this->size2_, ndim, axes, in_strides,
      in_shape, out_strides, out_shape, this->decay_rate_, this->eps_, x, gamma,
      beta, in_trans, m, v, rm, rv, y, mean_reduction_space,
      variance_reduction_space, inv_sqrt_variance);
#else
  forward_batch(this->size0_, this->size1_, this->size2_, this->decay_rate_,
                this->eps_, x, gamma, beta, m, v, rm, rv, y);
#endif
}

template <class T>
void BatchNormalizationCuda<T>::forward_impl_global(const Variables &inputs,
                                                    const Variables &outputs) {
  // Inputs
  const int b_idx = this->b_idx_;
  const int g_idx = this->g_idx_;
  const int m_idx = this->m_idx_;
  const int v_idx = this->v_idx_;
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *beta = this->no_bias_
                       ? nullptr
                       : inputs[b_idx]->get_data_pointer<Tc>(this->ctx_);
  const Tc *gamma = this->no_scale_
                        ? nullptr
                        : inputs[g_idx]->get_data_pointer<Tc>(this->ctx_);
  const Tc *rm =
      inputs[m_idx]->get_data_pointer<Tc>(this->ctx_); // running mean
  const Tc *rv = inputs[v_idx]->get_data_pointer<Tc>(this->ctx_); // running var
  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      forward_global_kernel, this->size1_ * this->size02_, this->size0_,
      this->size1_, this->size2_, this->size02_, this->size12_,
      this->decay_rate_, this->eps_, x, rm, rv, gamma, beta, y);
}

template <class T>
void BatchNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {

  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->batch_stat_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    this->backward_impl_global(inputs, outputs, propagate_down, accum);
  }
}

template <class T>
void BatchNormalizationCuda<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  const int b_idx = this->b_idx_;
  const int g_idx = this->g_idx_;
  const bool pd_beta = !this->no_bias_ && propagate_down[b_idx];
  const bool pd_gamma = !this->no_scale_ && propagate_down[g_idx];
  const bool accum_beta = !this->no_bias_ && accum[b_idx];
  const bool accum_gamma = !this->no_scale_ && accum[g_idx];
  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }
  // Common inputs wrt. gradient.
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *m = batch_mean->get_data_pointer<Tc>(this->ctx_);
  const Tc *v = batch_var->get_data_pointer<Tc>(this->ctx_);
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  auto get_data_ptr_ = [this](Variable &var) {
    return var.cast_data_and_get_pointer<Tc>(this->ctx_);
  };
#ifdef BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION
  int ndim = inputs[0]->ndim();
  auto get_ = [this](Variable &var) {
    return var.get_data_pointer<int>(this->ctx_);
  };
  const int *axes = get_(this->v_axes_);
  const int *in_strides = get_(this->v_in_strides_);
  const int *out_strides = get_(this->v_out_strides_);
  const int *in_shape = get_(this->v_in_shape_);
  const int *out_shape = get_(this->v_out_shape_);
  // TODO: write_only flags
  Tc *d_x_trans = get_data_ptr_(this->v_in_trans_);
  Tc *d_dy_trans = get_data_ptr_(this->v_din_trans_);
  Tc *mean_reduction_space = get_data_ptr_(this->v_mean_reduction_space_);
  Tc *variance_reduction_space =
      get_data_ptr_(this->v_variance_reduction_space_);
  Tc *inv_sqrt_variance = get_data_ptr_(this->v_inv_sqrt_variance_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      transpose_2value_kernel, this->size1_ * this->size02_, ndim, axes,
      in_strides, out_strides, out_shape, x, dy, d_x_trans, d_dy_trans);
#endif
  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero(); // TODO: optimize this out if possible
    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
    const Tc *g = this->no_scale_
                      ? nullptr
                      : inputs[g_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dm = nullptr;
    const Tc *dv = nullptr;
    if (outputs.size() == 3) {
      dm = batch_mean->get_grad_pointer<Tc>(this->ctx_);
      dv = batch_var->get_grad_pointer<Tc>(this->ctx_);
    }
    Tc *dmean = get_data_ptr_(this->v_dmean_);
    Tc *dvar = get_data_ptr_(this->v_dvar_);
#ifdef BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION
    Tc *tmp_reduction_space = get_data_ptr_(this->v_tmp_reduction_space_);
    Tc *t = get_data_ptr_(this->v_t_);
    backward_batch_data_parallel_reduction(
        this->size0_, this->size1_, this->size2_, ndim, axes, in_strides,
        in_shape, out_strides, out_shape, this->decay_rate_, this->eps_, dy, m,
        v, x, g, dm, dv, dx, mean_reduction_space, variance_reduction_space,
        tmp_reduction_space, dmean, dvar, t, inv_sqrt_variance, d_x_trans,
        d_dy_trans);
#else
    backward_batch_data(this->size0_, this->size1_, this->size2_,
                        this->decay_rate_, this->eps_, dy, m, v, x, g, dm, dv,
                        dx, dmean, dvar);
#endif
  }
  if (pd_beta || pd_gamma) { // beta and gamma
    if (!this->no_bias_ && !accum[b_idx])
      inputs[b_idx]->grad()->zero(); // TODO: optimize this out if possible
    if (!this->no_scale_ && !accum[g_idx])
      inputs[g_idx]->grad()->zero(); // TODO: optimize this out if possible
    Tc *db = !pd_beta ? nullptr
                      : inputs[b_idx]->cast_grad_and_get_pointer<Tc>(this->ctx_,
                                                                     false);
    Tc *dg =
        !pd_gamma
            ? nullptr
            : inputs[g_idx]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
#ifdef BATCH_NORMALIZATION_USE_PARALLEL_REDUCTION
    backward_batch_gamma_beta_parallel_reduction(
        this->size0_, this->size1_, this->size2_, d_dy_trans, m, v, d_x_trans,
        this->eps_, db, dg, mean_reduction_space, variance_reduction_space,
        inv_sqrt_variance);
#else
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward_batch_gamma_beta_kernel, this->size1_, this->size2_,
        this->size02_, this->size12_, this->eps_, dy, m, v, x, db, dg);
#endif
  }
}
}
