// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/solver/sgdw.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_update(const int num, T *data, const T *grad, T *v,
                              const float lr, const float momentum,
                              const float wd, T eta_t) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    v[idx] = momentum * v[idx] + lr * grad[idx] - (eta_t * wd * v[idx]);
    data[idx] -= v[idx];
  }
}

template <typename T>
void SgdWCuda<T>::update_impl(const string &key, VariablePtr param) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  VariablePtr r_ = state.pstate["m"];
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *v = r_->cast_data_and_get_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  T eta_t = this->lr_ / this->init_lr_;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_update, size, data, grad, v, this->lr_,
                                 this->momentum_, this->weight_decay_rate_,
                                 eta_t);
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(SgdWCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(SgdWCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(SgdWCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(SgdWCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(SgdWCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(SgdWCuda, scale_grad_impl_cuda);
}
