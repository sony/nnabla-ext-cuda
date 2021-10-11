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
#include <nbla/cuda/function/cumprod.hpp>
#include <nbla/variable.hpp>

#include <memory>

namespace nbla {

template <typename T>
void CumProdCuda<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  CumProd<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T, typename AccumType, bool exclusive, bool reverse>
__global__ void kernel_cumprod_forward(const int size0x2_, const int size1_,
                                       const int size2_, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;

    int j = i0 * size1_ * size2_ + i2;
    AccumType prod = (AccumType)1;
    for (int k = 0; k < size1_; ++k) {
      const int i1 = reverse ? size1_ - k - 1 : k;
      const int idx = i1 * size2_ + j;

      if (exclusive) {
        y[idx] = prod;
        prod *= x[idx];
      } else {
        prod *= x[idx];
        y[idx] = prod;
      }
    }
  }
}

template <typename T>
void CumProdCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto kernel = kernel_cumprod_forward<Tcu, AccumType, true /* exclusive */,
                                       true /* reverse */>;

  if (this->exclusive_) {
    kernel = this->reverse_
                 ? kernel_cumprod_forward<Tcu, AccumType, true, true>
                 : kernel_cumprod_forward<Tcu, AccumType, true, false>;
  } else {
    kernel = this->reverse_
                 ? kernel_cumprod_forward<Tcu, AccumType, false, true>
                 : kernel_cumprod_forward<Tcu, AccumType, false, false>;
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->size0_ * this->size2_,
                                 this->size1_, this->size2_, x, y);
}

template <typename T, typename AccumType, bool exclusive, bool reverse,
          bool accum>
__global__ void kernel_cumprod_backward(const int size0x2_, const int size1_,
                                        const int size2_, const T *x,
                                        const T *g_y, AccumType *masked_cumprod,
                                        T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;

    const int offset = i0 * size1_ * size2_ + i2;

    // Create masked_cumprod
    int first_zero_pos = size1_;
    AccumType prod = (AccumType)1;
    for (int k = 0; k < size1_; k++) {
      const int i1 = reverse ? size1_ - k - 1 : k;
      int idx = i1 * size2_ + offset;
      if (x[idx] == (T)0 && first_zero_pos == size1_) {
        first_zero_pos = k;
        // prod *= (AccumType)1;
      } else {
        prod *= x[idx];
      }
      masked_cumprod[k * size2_ + offset] = prod;
    }

    // Calculate gradient
    AccumType sum = 0;
    for (int k = size1_ - 1; k >= 0; k--) {
      const int i1 = reverse ? size1_ - k - 1 : k;
      int idx = i1 * size2_ + offset;

      if (!exclusive) {
        sum += masked_cumprod[k * size2_ + offset] * g_y[idx];
      }

      T grad;
      if (k == first_zero_pos) {
        grad = (T)sum;
        sum = 0;
      } else if (k > first_zero_pos) {
        grad = (T)0;
      } else {
        grad = (T)sum / x[idx];
      }
      g_x[idx] = grad + (accum ? g_x[idx] : (T)0);

      if (exclusive && k != 0) {
        sum += masked_cumprod[(k - 1) * size2_ + offset] * g_y[idx];
      }
    }
  }
}

template <typename T>
void CumProdCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);

  // `masked_cumprod` is a cumulative prod of `x` but treating the first zero
  // element as `1` on each `axis`.
  Variable v_masked_cumprod({inputs[0]->size()});
  AccumType *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<AccumType>(this->ctx_, true);

  size_t size = inputs[0]->size();

  auto kernel = kernel_cumprod_backward<Tcu, AccumType, true /* exclusive */,
                                        true /* reverse */, true /* accum */>;
  if (this->exclusive_) {
    if (this->reverse_) {
      kernel = accum[0]
                   ? kernel_cumprod_backward<Tcu, AccumType, true, true, true>
                   : kernel_cumprod_backward<Tcu, AccumType, true, true, false>;
    } else {
      kernel =
          accum[0]
              ? kernel_cumprod_backward<Tcu, AccumType, true, false, true>
              : kernel_cumprod_backward<Tcu, AccumType, true, false, false>;
    }
  } else {
    if (this->reverse_) {
      kernel =
          accum[0]
              ? kernel_cumprod_backward<Tcu, AccumType, false, true, true>
              : kernel_cumprod_backward<Tcu, AccumType, false, true, false>;
    } else {
      kernel =
          accum[0]
              ? kernel_cumprod_backward<Tcu, AccumType, false, false, true>
              : kernel_cumprod_backward<Tcu, AccumType, false, false, false>;
    }
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->size0_ * this->size2_,
                                 this->size1_, this->size2_, x, g_y,
                                 masked_cumprod, g_x);
}
}