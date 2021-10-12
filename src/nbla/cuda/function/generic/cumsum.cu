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

template <typename T, typename AccumType, bool exclusive, bool reverse>
__global__ void kernel_cumsum_forward(const int size0x2, const int size1,
                                      const int size2, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2) {
    const int i0 = idx / size2;
    const int i2 = idx % size2;

    int j = i0 * size1 * size2 + i2;
    AccumType sum = (AccumType)0;
    for (int k = 0; k < size1; ++k) {
      const int i1 = reverse ? size1 - k - 1 : k;
      const int idx = i1 * size2 + j;

      if (exclusive) {
        y[idx] = sum;
        sum += x[idx];
      } else {
        sum += x[idx];
        y[idx] = sum;
      }
    }
  }
}

template <typename T>
void CumSumCuda<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto kernel = kernel_cumsum_forward<Tcu, AccumType, true /* exclusive */,
                                      true /* reverse */>;

  if (this->exclusive_) {
    kernel = this->reverse_
                 ? kernel_cumsum_forward<Tcu, AccumType, true, true>
                 : kernel_cumsum_forward<Tcu, AccumType, true, false>;
  } else {
    kernel = this->reverse_
                 ? kernel_cumsum_forward<Tcu, AccumType, false, true>
                 : kernel_cumsum_forward<Tcu, AccumType, false, false>;
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->size0_ * this->size2_,
                                 this->size1_, this->size2_, x, y);
}

template <typename T, bool exclusive, bool reverse, bool accum>
__global__ void kernel_cumsum_backward(const int size0x2, const int size1,
                                       const int size2, const T *g_y, T *g_x) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2) {
    const int i0 = idx / size2;
    const int i2 = idx % size2;
    const int j = i0 * size1 * size2 + i2;

    AccumType cum_sum = T(0);
    for (int k = 0; k < size1; ++k) {

      const int i1 = reverse ? k : size1 - k - 1;
      const int idx = i1 * size2 + j;

      if (exclusive) {
        if (accum)
          g_x[idx] += cum_sum;
        else
          g_x[idx] = cum_sum;
        cum_sum += g_y[idx];
      } else {
        cum_sum += g_y[idx];
        if (accum)
          g_x[idx] += cum_sum;
        else
          g_x[idx] = cum_sum;
      }
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
  Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);

  auto kernel = kernel_cumsum_backward<Tcu, true /* exclusive */,
                                       true /* reverse */, true /* accum */>;

  if (this->exclusive_) {
    if (this->reverse_) {
      kernel = accum[0] ? kernel_cumsum_backward<Tcu, true, true, true>
                        : kernel_cumsum_backward<Tcu, true, true, false>;
    } else {
      kernel = accum[0] ? kernel_cumsum_backward<Tcu, true, false, true>
                        : kernel_cumsum_backward<Tcu, true, false, false>;
    }
  } else {
    if (this->reverse_) {
      kernel = accum[0] ? kernel_cumsum_backward<Tcu, false, true, true>
                        : kernel_cumsum_backward<Tcu, false, true, false>;
    } else {
      kernel = accum[0] ? kernel_cumsum_backward<Tcu, false, false, true>
                        : kernel_cumsum_backward<Tcu, false, false, false>;
    }
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->size0_ * this->size2_,
                                 this->size1_, this->size2_, g_y, g_x);
}
}