// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/solver/adam.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_adam_update(const int num, T *theta, T *m, T *v,
                                   const T *g, const float alpha_t,
                                   const float beta1, const float beta2,
                                   const float eps) {
  NBLA_CUDA_KERNEL_LOOP(s, num) {
    // Updating running mean and var.
    m[s] = beta1 * m[s] + (1 - beta1) * g[s];
    v[s] = beta2 * v[s] + (1 - beta2) * g[s] * g[s];
    // Update parameters.
    theta[s] = theta[s] - alpha_t * m[s] / (std::sqrt(v[s]) + eps);
  }
}

template <typename T>
void AdamCuda<T>::update_impl(const string &key, VariablePtr param) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  uint32_t &t = state.t;
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  shared_ptr<Variable> mean_ =
      state.pstate["mean"];                        // To prevent compile error.
  shared_ptr<Variable> var_ = state.pstate["var"]; // To prevent compile error.
  T *m = mean_->cast_data_and_get_pointer<T>(this->ctx_);
  T *v = var_->cast_data_and_get_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  const T bias_correction = std::sqrt(1 - std::pow(this->beta2_, t)) /
                            (1 - std::pow(this->beta1_, t));
  const T alpha_t = this->alpha_ * bias_correction;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_adam_update, size, theta, m, v, g,
                                 alpha_t, this->beta1_, this->beta2_,
                                 this->eps_);
}
NBLA_DEF_WEIGHT_DECAY(AdamCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdamCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(AdamCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(AdamCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdamCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(AdamCuda, scale_grad_impl_cuda);
}
