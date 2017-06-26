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
#include <nbla/cuda/solver/nesterov.hpp>

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
__global__ void kernel_weight_decay(const int num, T *grad, const T *data,
                                    const float decay_rate) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] += decay_rate * data[idx]; }
}

template <typename T>
void NesterovCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr v_ = this->state_.at(key);
  T *v = v_->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nesterov_update, size, data, grad, v,
                                 this->lr_, this->momentum_);
}

NBLA_DEF_WEIGHT_DECAY(NesterovCuda, weight_decay_cuda);

// Template instantiation
template class NesterovCuda<float>;
}
