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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/solver/adagrad.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_adagrad_update(const int num, T *data, const T *grad,
                                      T *g, const float lr, const float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    g[idx] += grad[idx] * grad[idx];
    data[idx] -= lr * grad[idx] / (sqrt(g[idx]) + eps);
  }
}

template <typename T>
void AdagradCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  VariablePtr g_ = state.pstate["v"];
  T *g = g_->cast_data_and_get_pointer<T>(this->ctx_);
  auto &t = state.t;
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_adagrad_update, size, data, grad, g,
                                 this->lr_, this->eps_);
}

NBLA_DEF_WEIGHT_DECAY(AdagradCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdagradCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(AdagradCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(AdagradCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdagradCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(AdagradCuda, scale_grad_impl_cuda);
}
