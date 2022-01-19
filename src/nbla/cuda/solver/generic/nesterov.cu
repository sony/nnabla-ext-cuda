// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/solver/nesterov.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_nesterov_update(const int num, T *data, const T *grad,
                                       T *v, const float lr,
                                       const float momentum) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T v_prev = v[idx];
    v[idx] = momentum * v[idx] - lr * grad[idx];
    data[idx] += -momentum * v_prev + (1 + momentum) * v[idx];
  }
}

template <typename T>
void NesterovCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  VariablePtr v_ = state.pstate["m"];
  T *v = v_->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nesterov_update, size, data, grad, v,
                                 this->lr_, this->momentum_);
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(NesterovCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(NesterovCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(NesterovCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(NesterovCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(NesterovCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(NesterovCuda, scale_grad_impl_cuda);
}
