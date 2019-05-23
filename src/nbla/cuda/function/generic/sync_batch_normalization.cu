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
#include <nbla/cuda/function/sync_batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

#include "kernel/batch_normalization.cu"

namespace nbla {

template <typename T>
void SyncBatchNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                               const Variables &outputs) {
  this->batch_norm_.setup(inputs, outputs);

  SyncBatchNormalization<T>::setup_impl(inputs, outputs);

  v_dmean_.reshape(Shape_t{this->size1_}, true);
  v_dvar_.reshape(Shape_t{this->size1_}, true);
  v_sync_.reshape(Shape_t{this->size1_ * 2}, true);
}

template <class T>
void SyncBatchNormalizationCuda<T>::forward_impl_batch(
    const Variables &inputs, const Variables &outputs) {
  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }
  // Inputs
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *beta = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  const Tc *gamma = inputs[2]->get_data_pointer<Tc>(this->ctx_);
  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Tc *m =
      batch_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true); // batch mean
  Tc *v =
      batch_var->cast_data_and_get_pointer<Tc>(this->ctx_, true); // batch varf
  // Inputs/Outputs
  Tc *rm = inputs[3]->cast_data_and_get_pointer<Tc>(this->ctx_); // running mean
  Tc *rv = inputs[4]->cast_data_and_get_pointer<Tc>(this->ctx_); // running var

  // Calculate mean and squared-mean
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_mean_sqmean_kernel,
                                 /* Input */
                                 this->size1_, this->size2_,
                                 this->size0_ * this->size2_,
                                 this->size1_ * this->size2_, x,
                                 /* Output */
                                 m, v);

  // Sync between other processes
  this->comm_->all_reduce({batch_mean->data(), batch_var->data()}, false, false,
                          this->group_);

  m = batch_mean->cast_data_and_get_pointer<Tc>(this->ctx_); // batch mean
  v = batch_var->cast_data_and_get_pointer<Tc>(this->ctx_);  // batch varf
  // Calculate running mean and var
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_running_mean_var_kernel,
                                 /* Input */
                                 this->size1_, this->size0_ * this->size2_,
                                 this->num_processes_, this->decay_rate_, m, v,
                                 /* Output */
                                 rm, rv);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      forward_batch_gamma_beta_kernel,
      /* Input */
      this->size1_ * this->size0_ * this->size2_, this->size0_, this->size2_,
      this->size0_ * this->size2_, this->size1_ * this->size2_,
      this->decay_rate_, this->eps_, x, m, v, rm, rv, gamma, beta,
      /* Output */
      y);
}

template <class T>
void SyncBatchNormalizationCuda<T>::forward_impl_global(
    const Variables &inputs, const Variables &outputs) {
  this->batch_norm_.forward(inputs, outputs);
}

template <class T>
void SyncBatchNormalizationCuda<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
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
  const Tc *g = inputs[2]->get_data_pointer<Tc>(this->ctx_);
  const Tc *dm = nullptr;
  const Tc *dv = nullptr;
  if (outputs.size() == 3) {
    dm = batch_mean->get_grad_pointer<Tc>(this->ctx_);
    dv = batch_var->get_grad_pointer<Tc>(this->ctx_);
  }
  auto get_data_ptr_ = [this](Variable &var) {
    return var.cast_data_and_get_pointer<Tc>(this->ctx_);
  };

  // Synchronize between other processes
  Tc *buff = get_data_ptr_(this->v_sync_);
  Tc *sum_dy_ptr = buff;
  Tc *sum_xdy_ptr = buff + this->size1_;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_batch_data_pre_sync_kernel,
                                 /* Input */
                                 this->size1_, this->size2_,
                                 this->size0_ * this->size2_,
                                 this->size1_ * this->size2_, this->decay_rate_,
                                 this->eps_, dy, m, v, x, g, dm, dv,
                                 /* Output */
                                 sum_dy_ptr, sum_xdy_ptr);
  // Sync between other processes
  this->comm_->all_reduce(this->v_sync_.data(), false, false, this->group_);
  buff = get_data_ptr_(this->v_sync_);
  sum_dy_ptr = buff;
  sum_xdy_ptr = buff + this->size1_;

  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero(); // TODO: optimize this out if possible
    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
    Tc *dmean = get_data_ptr_(this->v_dmean_);
    Tc *dvar = get_data_ptr_(this->v_dvar_);

    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward_batch_data_mean_variance_post_sync_kernel,
        /* Input */
        this->size1_, this->size0_ * this->size2_, this->eps_, m, v, g, dm, dv,
        sum_dy_ptr, sum_xdy_ptr,
        /* Output */
        dmean, dvar);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward_batch_data_dx_post_sync_kernel,
        /* Input */
        this->size1_ * this->size0_ * this->size2_, this->size0_, this->size1_,
        this->size2_, this->size0_ * this->size2_, this->size1_ * this->size2_,
        this->num_processes_ * this->size02_, this->decay_rate_, this->eps_, dy,
        m, v, x, g, dm, dv, dmean, dvar,
        /* Output */
        dx);
  }
  if (propagate_down[1] || propagate_down[2]) { // beta and gamma
    NBLA_CHECK(propagate_down[1] && propagate_down[2], error_code::value,
               "'need_grad' of beta and gamma must be the same.");
    if (!accum[1])
      inputs[1]->grad()->zero(); // TODO: optimize this out if possible
    if (!accum[2])
      inputs[2]->grad()->zero(); // TODO: optimize this out if possible
    Tc *db = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
    Tc *dg = inputs[2]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_batch_gamma_beta_post_sync_kernel,
                                   this->size1_, this->size2_, this->size02_,
                                   this->size12_, this->eps_, dy, m, v, x,
                                   sum_dy_ptr, sum_xdy_ptr, db, dg);
  }
}
}
