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
#include <nbla/cuda/function/cumsum.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void CumSumCuda<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  CumSum<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T, typename AccumType>
__global__ void kernel_cumsum_forward(const int size0x2_, const int size1_,
                                      const int size2_, const T *x,
                                      AccumType *y, bool exclusive_,
                                      bool reverse_) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;

    int j = i0 * size1_ * size2_ + i2;
    for (int index = 0; index < size1_; ++index) {
      const int i1 = reverse_ ? size1_ - index - 1 : index;

      const int d = reverse_ ? -1 : 1;
      const int x_k = exclusive_ ? (i1 - d) * size2_ + j : i1 * size2_ + j;
      const int y_k = i1 * size2_ + j;
      const int y_k_prev = y_k - d * size2_;

      y[y_k] = index != 0 ? y[y_k_prev] + x[x_k] : exclusive_ ? 0 : x[x_k];
    }
  }
}

template <typename T>
void CumSumCuda<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  AccumType *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_cumsum_forward, this->size0_ * this->size2_, this->size1_,
      this->size2_, x, y, this->exclusive_, this->reverse_);
}

template <typename T>
__global__ void kernel_cumsum_backward(const int size0x2_, const int size1_,
                                       const int size2_, const T *g_y, T *g_x,
                                       bool exclusive_, bool reverse_,
                                       bool accum) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size1_ * size2_ + i2;

    AccumType cum_sum = T(0);
    for (int index = 0; index < size1_; ++index) {

      const int i1 = reverse_ ? index : size1_ - index - 1;
      const int x_k = i1 * size2_ + j;

      cum_sum += g_y[x_k];
      if (accum)
        g_x[x_k] += (exclusive_ ? cum_sum - g_y[x_k] : cum_sum);
      else
        g_x[x_k] = (exclusive_ ? cum_sum - g_y[x_k] : cum_sum);
    }
  }
}

template <typename T>
void CumSumCuda<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  if (propagate_down[0]) {
    Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_cumsum_backward<Tcu>),
                                   this->size0_ * this->size2_, this->size1_,
                                   this->size2_, g_y, g_x, this->exclusive_,
                                   this->reverse_, accum[0]);
  }
}
}