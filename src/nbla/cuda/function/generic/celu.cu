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
#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/celu.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_celu_forward(const int size10_, const int size0_,
                                    const float alpha, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size10_) {
    int i1 = idx / size0_;
    int i0 = idx % size0_;
    const int j0 = i1 * size0_ * 2 + i0;
    const T &xk = x[i1 * size0_ + i0];
    y[j0] = 0 <= xk ? xk : (T)alpha * (std::exp(xk) - 1);
    y[j0 + size0_] = xk <= 0 ? -xk : (T)alpha * (std::exp(-xk) - 1);
  }
}

template <typename T, bool accum>
__global__ void kernel_celu_backward(const int size10_, const int size0_,
                                     const float alpha, const T *x, const T *dy,
                                     T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size10_) {
    int i1 = idx / size0_;
    int i0 = idx % size0_;
    const int j0 = i1 * size0_ * 2 + i0;
    const int j1 = j0 + size0_;
    const int k = i1 * size0_ + i0;
    const T &dyj0 = dy[j0];
    const T &dyj1 = dy[j1];
    const T &xk = x[k];
    dx[k] = (accum ? dx[k] : (T)0) +
            (0 <= xk ? dyj0 : dyj0 * (T)alpha * std::exp(xk));
    dx[k] -= xk <= 0 ? dyj1 : dyj1 * (T)alpha * std::exp(-xk);
  }
}

template <typename T>
void CELUCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  CELU<T>::setup_impl(inputs, outputs);
}

template <typename T>
void CELUCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_celu_forward,
                                 this->size0_ * this->size1_, this->size0_,
                                 this->alpha_, x, y);
}

template <typename T>
void CELUCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_celu_backward<Tc, true>),
                                   this->size0_ * this->size1_, this->size0_,
                                   this->alpha_, x, dy, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_celu_backward<Tc, false>),
                                   this->size0_ * this->size1_, this->size0_,
                                   this->alpha_, x, dy, dx);
  }
}
}
