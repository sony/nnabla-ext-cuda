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

// relu.cpp

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/relu.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_relu_forward(const int num, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = max(T(0), x[idx]); }
}

template <typename T, bool accum = true>
__global__ void kernel_relu_backward(const int num, T *dx, const T *x,
                                     const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : 0) + (x[idx] > 0 ? dy[idx] : 0);
  }
}

template <class T>
void ReLUCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_relu_forward, size, y, x);
}

template <class T>
void ReLUCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  size_t size = inputs[0]->size();
  if (dx != dy && accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_relu_backward<T, true>), size, dx, x,
                                   dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_relu_backward<T, false>), size, dx,
                                   x, dy);
  }
}

// Template instantiation
template class ReLUCuda<float>;
}
