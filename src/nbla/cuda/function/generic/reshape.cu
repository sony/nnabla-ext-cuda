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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/reshape.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_reshape_forward(const int num, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x[idx]; }
}

template <typename T, bool accum = true>
__global__ void kernel_reshape_backward(const int num, T *dx, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    if (accum) {
      dx[idx] += dy[idx];
    } else {
      dx[idx] = dy[idx];
    }
  }
}

template <typename T>
void ReshapeCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  if (this->inplace_) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reshape_forward, size, y, x);
}

template <typename T>
void ReshapeCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(
      this->ctx_, !(this->inplace_ || accum[0]));
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  size_t size = inputs[0]->size();
  if (dx != dy && accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reshape_backward<Tcu, true>), size,
                                   dx, dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reshape_backward<Tcu, false>), size,
                                   dx, dy);
  }
}
}
