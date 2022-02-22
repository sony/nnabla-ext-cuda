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

// crelu.cpp

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/crelu.hpp>

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_crelu_forward(const int size10_, const int size0_,
                                     const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size10_) {
    int i1 = idx / size0_;
    int i0 = idx % size0_;
    y[i1 * size0_ * 2 + i0] = max(T(0), x[i1 * size0_ + i0]);
    y[i1 * size0_ * 2 + size0_ + i0] = max(T(0), -x[i1 * size0_ + i0]);
  }
}

template <typename T, bool accum>
__global__ void kernel_crelu_backward(const int size10_, const int size0_,
                                      const T *x, const T *dy, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size10_) {
    int i1 = idx / size0_;
    int i0 = idx % size0_;
    dx[i1 * size0_ + i0] =
        (accum ? dx[i1 * size0_ + i0] : (T)0) +
        (x[i1 * size0_ + i0] > 0
             ? dy[i1 * size0_ * 2 + i0]
             : -(x[i1 * size0_ + i0] < 0 ? dy[i1 * size0_ * 2 + size0_ + i0]
                                         : (T)0));
  }
}

template <typename T>
void CReLUCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  CReLU<T>::setup_impl(inputs, outputs);
}

template <class T>
void CReLUCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_crelu_forward, this->size0_ * this->size1_, this->size0_, x, y);
}

template <class T>
void CReLUCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_crelu_backward<Tc, true>),
                                   this->size0_ * this->size1_, this->size0_, x,
                                   dy, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_crelu_backward<Tc, false>),
                                   this->size0_ * this->size1_, this->size0_, x,
                                   dy, dx);
  }
}
}
