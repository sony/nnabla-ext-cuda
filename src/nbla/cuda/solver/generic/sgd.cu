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
#include <nbla/cuda/solver/sgd.hpp>

#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_update(const int num, T *data, const T *grad,
                              const float lr) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { data[idx] -= lr * grad[idx]; }
}

template <typename T>
void SgdCuda<T>::update_impl(const string &key, VariablePtr param) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Size_t size = param->size();
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_update, size, data, grad, this->lr_);
}

NBLA_DEF_WEIGHT_DECAY(SgdCuda, weight_decay_cuda);
}
