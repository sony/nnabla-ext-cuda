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

// binary_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/binary_cross_entropy.hpp>
#include <nbla/cuda/limits.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_binary_cross_entropy_forward(const int size, const T *x0,
                                                    const T *x1, T *y) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    y[s] = -(x1[s] * log(max(x0[s], numeric_limits_cuda<T>::min())) +
             (1 - x1[s]) * log(max(1 - x0[s], numeric_limits_cuda<T>::min())));
  }
}

template <typename T, bool accum>
__global__ void
kernel_binary_cross_entropy_backward_dx0(const int size, const T *dy,
                                         const T *x0, const T *x1, T *dx0) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    dx0[s] = (accum ? dx0[s] : (T)0) +
             dy[s] * (x0[s] - x1[s]) /
                 max(x0[s] - x0[s] * x0[s], numeric_limits_cuda<T>::min());
  }
}

template <typename T, bool accum>
__global__ void
kernel_binary_cross_entropy_backward_dx1(const int size, const T *dy,
                                         const T *x0, const T *x1, T *dx1) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    dx1[s] = (accum ? dx1[s] : (T)0) +
             dy[s] * (log(max(1 - x0[s], numeric_limits_cuda<T>::min())) -
                      log(max(x0[s], numeric_limits_cuda<T>::min())));
  }
}

template <typename T>
void BinaryCrossEntropyCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  BinaryCrossEntropy<T>::setup_impl(inputs, outputs);
}

template <typename T>
void BinaryCrossEntropyCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x0 = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *x1 = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_binary_cross_entropy_forward, size, x0,
                                 x1, y);
}

template <typename T>
void BinaryCrossEntropyCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *x0 = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *x1 = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  const Size_t size = inputs[0]->size();
  if (propagate_down[0]) {
    Tc *dx0 = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_binary_cross_entropy_backward_dx0<Tc, true>), size, dy, x0,
          x1, dx0);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_binary_cross_entropy_backward_dx0<Tc, false>), size, dy, x0,
          x1, dx0);
    }
  }
  if (propagate_down[1]) {
    Tc *dx1 = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[1]);
    if (accum[1]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_binary_cross_entropy_backward_dx1<Tc, true>), size, dy, x0,
          x1, dx1);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_binary_cross_entropy_backward_dx1<Tc, false>), size, dy, x0,
          x1, dx1);
    }
  }
}
}
