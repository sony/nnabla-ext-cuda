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
#include <nbla/cuda/function/clip_grad_by_value.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_clip_grad_by_value_forward(const int num, T *y,
                                                  const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x[idx]; }
}

template <typename T, bool accum = true>
__global__ void kernel_clip_grad_by_value_backward(const int num, T *dx,
                                                   const T *dy, const T *min,
                                                   const T *max) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T min_i = min[idx];
    T max_i = max[idx];
    T value;
    if (dy[idx] > max_i) {
      value = max_i;
    } else if (dy[idx] < min_i) {
      value = min_i;
    } else {
      value = dy[idx];
    }
    if (accum) {
      dx[idx] += value;
    } else {
      dx[idx] = value;
    }
  }
}

template <typename T>
void ClipGradByValueCuda<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  cuda_set_device(this->device_);
  Size_t size = inputs[0]->size();
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_clip_grad_by_value_forward, size, y, x);
}

template <typename T>
void ClipGradByValueCuda<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  cuda_set_device(this->device_);
  // No backward to min and max variables.
  if (!propagate_down[0]) {
    return;
  }

  // Zeroing grads of min and max when accum is false.
  for (int i = 1; i < 3; i++) {
    if (propagate_down[i] && !accum[i]) {
      inputs[i]->grad()->zero();
    }
  }

  Size_t size = inputs[0]->size();
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *min = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  const Tc *max = inputs[2]->get_data_pointer<Tc>(this->ctx_);

  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_clip_grad_by_value_backward<Tc, true>), size, dx, dy, min, max);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_clip_grad_by_value_backward<Tc, false>), size, dx, dy, min,
        max);
  }
}
}
