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

// leaky_relu.cu

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/leaky_relu.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_leaky_relu_forward(const int num, T *y, const T *x,
                                          float alpha) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T x_idx = x[idx];
    if (x_idx > 0) {
      y[idx] = x_idx;
    } else {
      y[idx] = alpha * x_idx;
    }
  }
}

template <typename T, bool accum = true>
__global__ void kernel_leaky_relu_backward(const int num, T *dx, const T *sign,
                                           const T *dy, float alpha) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    if (accum) {
      if (sign[idx] > 0)
        dx[idx] += dy[idx];
      else
        dx[idx] += alpha * dy[idx];
    } else {
      if (sign[idx] > 0)
        dx[idx] = dy[idx];
      else
        dx[idx] = alpha * dy[idx];
    }
  }
}

template <class T>
void LeakyReLUCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_leaky_relu_forward, size, y, x,
                                 this->alpha_);
}

template <class T>
void LeakyReLUCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *sign = (this->alpha_ >= 0)
                       ? outputs[0]->get_data_pointer<Tc>(this->ctx_)
                       : inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  size_t size = inputs[0]->size();
  if (dx != dy && accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_leaky_relu_backward<Tc, true>), size,
                                   dx, sign, dy, this->alpha_);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_leaky_relu_backward<Tc, false>),
                                   size, dx, sign, dy, this->alpha_);
  }
}
}
