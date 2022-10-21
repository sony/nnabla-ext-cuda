// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/sync_batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/concatenate.hpp>
#include <nbla/function/slice.hpp>

#include "kernel/sync_batch_normalization.cu"

namespace nbla {

template <typename T>
void SyncBatchNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                               const Variables &outputs) {
  this->batch_norm_.setup(inputs, outputs);

  SyncBatchNormalization<T>::setup_impl(inputs, outputs);

  int c = this->size1_;

  //----------------
  // Forward setup
  //----------------
  v_local_mean_.reshape(Shape_t{c}, true);
  v_local_invstd_.reshape(Shape_t{c}, true);
  v_local_count_.reshape(Shape_t{1}, true);

  // Store local reduction size to the buffer in order to synchronize between
  // other processes and get total reduction size.
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  float *local_count = v_local_count_.cast_data_and_get_pointer<float>(cpu_ctx);
  local_count[0] = this->size02_;

  int n_workers = this->comm_->size();
  v_all_gather_send_.reshape(Shape_t{2 * c + 1}, true);
  v_all_gather_recv_.reshape(Shape_t{n_workers, (2 * c + 1)}, true);

  // Concatenate buffers for all_gather.
  concatenate_ = create_Concatenate(this->ctx_, 0);
  concatenate_->setup({&v_local_mean_, &v_local_invstd_, &v_local_count_},
                      {&v_all_gather_send_});

  // Slices for extracting the result of all_gather.
  slice_mean_ = create_Slice(this->ctx_, {0, 0}, {n_workers, c}, {1, 1});
  slice_mean_->setup({&v_all_gather_recv_}, {&v_all_mean_});

  slice_invstd_ = create_Slice(this->ctx_, {0, c}, {n_workers, 2 * c}, {1, 1});
  slice_invstd_->setup({&v_all_gather_recv_}, {&v_all_invstd_});

  slice_count_ =
      create_Slice(this->ctx_, {0, 2 * c}, {n_workers, 2 * c + 1}, {1, 1});
  slice_count_->setup({&v_all_gather_recv_}, {&v_all_count_});

  //----------------
  // Backward setup
  //----------------
  v_sum_dy_o_.reshape(Shape_t{c}, true);
  v_sum_dy_xmu_o_.reshape(Shape_t{c}, true);

  v_beta_grad_.reshape(inputs[1]->shape(), true);
  v_gamma_grad_.reshape(inputs[2]->shape(), true);
  add2_ = create_Add2(this->ctx_, true /* inplace */);
}

template <class T>
void SyncBatchNormalizationCuda<T>::forward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const bool update_inputs) {

  Variable *x = inputs[0];
  Variable *beta = inputs[1];
  Variable *gamma = inputs[2];
  Variable *y = outputs[0];

  const Size_t size0 = this->size0_;
  const Size_t size1 = this->size1_;
  const Size_t size2 = this->size2_;

  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }

  const bool channel_last = this->axes_[0] == inputs[0]->ndim() - 1;

  // Calculate local mean and variance
  if (channel_last) {
    forward_collect_statistics_channels_last<Tc>(
        size0, size1, size2, x, &v_local_mean_, &v_local_invstd_,
        &v_staging_data_for_forward_, &v_semaphores_for_forward_, this->eps_,
        this->ctx_);
  } else {
    forward_collect_statistics<Tc>(size0, size1, size2, x, &v_local_mean_,
                                   &v_local_invstd_, this->eps_, this->ctx_);
  }

  // All gather local mean, variance and count
  concatenate_->forward({&v_local_mean_, &v_local_invstd_, &v_local_count_},
                        {&v_all_gather_send_});
  const auto send_buffer = v_all_gather_send_.data();
  const auto recv_buffer = v_all_gather_recv_.data();
  this->comm_->all_gather(send_buffer, {recv_buffer}, this->group_);

  // Calculate global mean, variance
  slice_mean_->forward({&v_all_gather_recv_}, {&v_all_mean_});
  slice_invstd_->forward({&v_all_gather_recv_}, {&v_all_invstd_});
  slice_count_->forward({&v_all_gather_recv_}, {&v_all_count_});
  auto r_mean = !update_inputs ? nullptr : inputs[3];
  auto r_var = !update_inputs ? nullptr : inputs[4];
  const int n_workers = this->comm_->size();
  forward_reduce_statistics<Tc>(size0, size1, size2, &v_all_mean_,
                                &v_all_invstd_, &v_all_count_, batch_mean,
                                batch_var, r_mean, r_var, this->eps_,
                                this->decay_rate_, this->ctx_, n_workers);

  // Batch normalization
  if (channel_last) {
    forward_normalization_channel_last<Tc>(size0, size1, size2, x, batch_mean,
                                           batch_var, beta, gamma, y,
                                           this->eps_, this->ctx_);
  } else {
    forward_normalization<Tc>(size0, size1, size2, x, y, batch_mean, batch_var,
                              beta, gamma, this->eps_, this->ctx_);
  }

  // Clear internal buffers used only forward.
  {
    v_staging_data_for_forward_.data()->array()->clear();
    v_semaphores_for_forward_.data()->array()->clear();
    v_local_mean_.data()->array()->clear();
    v_local_invstd_.data()->array()->clear();
    // v_local_count is constant value with size 1. This value is set by CPU
    // context in setup_impl and cannot be clear during propagation.
    // v_local_count_.data()->array()->clear();
    v_all_gather_send_.data()->array()->clear();
    v_all_gather_recv_.data()->array()->clear();
    v_all_mean_.data()->array()->clear();
    v_all_invstd_.data()->array()->clear();
    // v_all_count_ will be used in backward
    // v_all_count_.data()->array()->clear();
  }
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

  Variable *x = inputs[0];
  Variable *beta = inputs[1];
  Variable *gamma = inputs[2];
  Variable *y = outputs[0];

  const Size_t size0 = this->size0_;
  const Size_t size1 = this->size1_;
  const Size_t size2 = this->size2_;

  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }

  const bool channel_last = this->axes_[0] == inputs[0]->ndim() - 1;

  // Reduce channels and calculate grad of beta and gamma to temporally buffers
  if (channel_last) {
    backward_reduce_channels_last<Tc>(
        size0, size1, size2, x, y, batch_mean, batch_var, &v_sum_dy_o_,
        &v_sum_dy_xmu_o_, &v_beta_grad_, &v_gamma_grad_,
        &v_staging_data_for_backward_, &v_semaphores_for_backward_, this->eps_,
        this->ctx_);
  } else {
    backward_reduce<Tc>(size0, size1, size2, x, y, batch_mean, batch_var,
                        &v_sum_dy_o_, &v_sum_dy_xmu_o_, &v_beta_grad_,
                        &v_gamma_grad_, this->eps_, this->ctx_);
  }

  // All reduce
  this->comm_->all_reduce({v_sum_dy_o_.data(), v_sum_dy_xmu_o_.data()}, false,
                          false, this->group_);

  // Store beta grad and gamma grad
  auto set_param_grad = [&](Variable *param, Variable *param_grad_global,
                            const bool accum) {
    if (accum) {
      // Copy current grad value.
      Variable tmp_var(param->grad());
      // Accumulate gradient by add2 operation.
      nbla::execute(add2_, {&tmp_var, param_grad_global}, {&tmp_var});
    } else {
      // Just copy grad.
      const Array *param_from =
          param_grad_global->data()->get(get_dtype<T>(), this->ctx_);
      Array *param_to = param->grad()->cast(get_dtype<T>(), this->ctx_, true);
      param_to->copy_from(param_from);
    }
  };
  // Beta grad
  if (propagate_down[1]) {
    set_param_grad(beta, &v_beta_grad_, accum[1]);
  }
  // Gamma grad
  if (propagate_down[2]) {
    set_param_grad(gamma, &v_gamma_grad_, accum[2]);
  }

  // Calculate x grad
  if (propagate_down[0]) {
    const bool output_stat = outputs.size() == 3;
    const int n_workers = this->comm_->size();
    if (channel_last) {
      if (accum[0]) {
        backward_dx_post_channels_last<Tc, true>(
            size0, size1, size2, y, x, batch_mean, batch_var, gamma,
            &v_sum_dy_o_, &v_sum_dy_xmu_o_, &v_all_count_, output_stat,
            this->eps_, this->ctx_);
      } else {
        backward_dx_post_channels_last<Tc, false>(
            size0, size1, size2, y, x, batch_mean, batch_var, gamma,
            &v_sum_dy_o_, &v_sum_dy_xmu_o_, &v_all_count_, output_stat,
            this->eps_, this->ctx_);
      }
    } else {
      if (accum[0]) {
        backward_dx_post<Tc, true>(size0, size1, size2, x, y, batch_mean,
                                   batch_var, &v_sum_dy_o_, &v_sum_dy_xmu_o_,
                                   gamma, &v_all_count_, output_stat,
                                   this->eps_, this->ctx_);
      } else {
        backward_dx_post<Tc, false>(size0, size1, size2, x, y, batch_mean,
                                    batch_var, &v_sum_dy_o_, &v_sum_dy_xmu_o_,
                                    gamma, &v_all_count_, output_stat,
                                    this->eps_, this->ctx_);
      }
    }
  }

  // Clear internal buffers used in backward.
  {
    v_staging_data_for_backward_.data()->array()->clear();
    v_semaphores_for_backward_.data()->array()->clear();
    v_sum_dy_o_.data()->array()->clear();
    v_sum_dy_xmu_o_.data()->array()->clear();
    v_beta_grad_.data()->array()->clear();
    v_gamma_grad_.data()->array()->clear();
    v_all_count_.data()->array()->clear(); // Calculated in forwrad
  }
}
}
