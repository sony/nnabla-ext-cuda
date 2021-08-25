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

template <typename T, typename AccumType>
__global__ void kernel_cumprod_forward(const int size0x2_, const int size1_,
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

      y[y_k] = index != 0 ? y[y_k_prev] * x[x_k] : exclusive_ ? 1 : x[x_k];
    }
  }
}

template <typename T>
void CumProdCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  AccumType *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_cumprod_forward, this->size0_ * this->size2_, this->size1_,
      this->size2_, x, y, this->exclusive_, this->reverse_);
}

template <typename T>
__global__ void
kernel_cumprod_backward(const int size0x2_, const int size1_, const int size2_,
                        const T *x, const T *y, const T *g_y, T *g_x,
                        bool exclusive_, bool reverse_, bool accum) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size1_ * size2_ + i2;

    AccumType cum_sum_ydy = T(0);
    for (int index = 0; index < size1_; ++index) {

      const int i1 = reverse_ ? index : size1_ - index - 1;
      const int x_k = i1 * size2_ + j;

      cum_sum_ydy += y[x_k] * g_y[x_k];
      if (accum) {
        g_x[x_k] +=
            (exclusive_ ? cum_sum_ydy - y[x_k] * g_y[x_k] : cum_sum_ydy) /
            x[x_k];
      } else {
        g_x[x_k] =
            (exclusive_ ? cum_sum_ydy - y[x_k] * g_y[x_k] : cum_sum_ydy) /
            x[x_k];
      }
    }
  }
}

template <typename T>
__global__ void
kernel_cumprod_backward_zero_input(const int size0x2_, const int size1_,
                                   const int size2_, const T *x, const T *y,
                                   const T *g_y, T *g_x, bool exclusive_,
                                   bool reverse_, bool accum) {
  typedef typename CudaTypeForceFloat<T>::type AccumType;
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size1_ * size2_ + i2;

    AccumType cur = T(0);
    for (int index = 0; index < size1_; ++index) {

      const int i1 = reverse_ ? index : size1_ - index - 1;
      const int x_k = i1 * size2_ + j;

      T coeff =
          (i1 == 0) ? (T)1 : exclusive_ ? y[x_k] : y[(i1 - 1) * size2_ + j];
      if (reverse_)
        coeff = (i1 == size1_ - 1) ? (T)1 : exclusive_
                                                ? y[x_k]
                                                : y[(i1 + 1) * size2_ + j];

      cur = exclusive_ ? (T)0 : coeff * g_y[x_k];

      if (reverse_) {
        for (int i4 = i1 - 1; i4 >= 0; --i4) {
          if (!exclusive_ || i4 != i1 - 1)
            coeff *=
                (exclusive_ ? x[(i4 + 1) * size2_ + j] : x[i4 * size2_ + j]);
          cur += coeff * g_y[i4 * size2_ + j];
        }
      } else {
        for (int i4 = i1 + 1; i4 < size1_; ++i4) {
          if (!exclusive_ || i4 != i1 + 1) {
            coeff *=
                (exclusive_ ? x[(i4 - 1) * size2_ + j] : x[i4 * size2_ + j]);
          }
          cur += coeff * g_y[i4 * size2_ + j];
        }
      }

      if (accum)
        g_x[x_k] += cur;
      else
        g_x[x_k] = cur;
    }
  }
}

template <typename T>
__global__ void kernel_zero_input_check(const int size0x2_, const T *x,
                                        bool *zero_input_present) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    if (x[idx] == (T)0) {
      *zero_input_present = true;
      break;
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
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);

  bool *d_zero_input_present, h_zero_input_present = false;
  NBLA_CUDA_CHECK(cudaMalloc(&d_zero_input_present, sizeof(bool)));
  NBLA_CUDA_CHECK(cudaMemcpy(d_zero_input_present, &h_zero_input_present,
                             sizeof(bool), cudaMemcpyHostToDevice));

  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_zero_input_check<Tcu>), size, x,
                                 d_zero_input_present);
  NBLA_CUDA_CHECK(cudaMemcpy(&h_zero_input_present, d_zero_input_present,
                             sizeof(bool), cudaMemcpyDeviceToHost));

  if (propagate_down[0]) {
    Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);

    if (h_zero_input_present == 1) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_cumprod_backward_zero_input<Tcu>),
          this->size0_ * this->size2_, this->size1_, this->size2_, x, y, g_y,
          g_x, this->exclusive_, this->reverse_, accum[0]);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_cumprod_backward<Tcu>), this->size0_ * this->size2_,
          this->size1_, this->size2_, x, y, g_y, g_x, this->exclusive_,
          this->reverse_, accum[0]);
    }
  }
}
}