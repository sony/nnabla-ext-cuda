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

// softmax.cu

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/softmax.hpp>
#include <nbla/cuda/limits.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_softmax_forward(const int size0x2_, const int size1_,
                                       const int size2_, const T *x, T *y) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    // compute maximum
    AccumType max_x = -nbla::numeric_limits_cuda<T>::max();
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      max_x = max(max_x, x[k]);
    }
    // Compute exponential and sum
    AccumType exp_sum = T(0);
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      const AccumType tmp = std::exp(x[k] - max_x);
      y[k] = tmp;
      exp_sum += tmp;
    }
    // Compute softmax
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      y[k] = y[k] / exp_sum;
    }
  }
}

template <typename T, bool accum>
__global__ void kernel_softmax_backward(const int size0x2_, const int size1_,
                                        const int size2_, const T *y,
                                        const T *dy, T *dx) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    // compute sum of dy * y
    AccumType dyy_sum = T(0);
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      dyy_sum += dy[k] * y[k];
    }
    // Compute backward
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = (i0 * size1_ + i1) * size2_ + i2;
      dx[k] = (accum ? dx[k] : (T)0) + y[k] * (dy[k] - dyy_sum);
    }
  }
}

template <class T>
void SoftmaxCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Setting up variables
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_softmax_forward,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, x, y);
}

template <class T>
void SoftmaxCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Setting up variables
  const Tc *y = outputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_softmax_backward<Tc, true>),
                                   this->size0_ * this->size2_, this->size1_,
                                   this->size2_, y, dy, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_softmax_backward<Tc, false>),
                                   this->size0_ * this->size2_, this->size1_,
                                   this->size2_, y, dy, dx);
  }
}
}
