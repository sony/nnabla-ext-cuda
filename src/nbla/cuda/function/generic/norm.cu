// Copyright 2020,2021 Sony Corporation.
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

#include <cmath>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/norm.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_abs_pow(const int size, const T *x, T *y,
                               const float p) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = std::pow(abs(x[idx]), (T)p); }
}

template <typename T>
__global__ void kernel_pow(const int size, const T *x, T *y, const float p) {
  // optimization with replacing pow to sqrt
  if (p == 0.5f) {
    NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = sqrt(x[idx]); }
  } else {
    NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = std::pow(x[idx], (T)p); }
  }
}

template <typename T, bool accum>
__global__ void kernel_abs_pow_backward(const int size, const T *x, const T *gy,
                                        T *gx, const float p) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    if (x[idx] < 0) {
      gx[idx] = (accum ? gx[idx] : (T)0.) +
                -p * std::pow(abs(x[idx]), (T)(p - 1.)) * gy[idx];
    } else {
      gx[idx] = (accum ? gx[idx] : (T)0.) +
                p * std::pow(abs(x[idx]), (T)(p - 1.)) * gy[idx];
    }
  }
}

template <typename T>
__global__ void kernel_pow_backward(const int size, const T *x, const T *gy,
                                    T *gx, const float p) {
  // optimization with replacing pow to rsqrt
  if (p == 0.5f) {
    NBLA_CUDA_KERNEL_LOOP(idx, size) {
      gx[idx] = (T)0.5 * rsqrt(x[idx]) * gy[idx];
    }
  } else {
    NBLA_CUDA_KERNEL_LOOP(idx, size) {
      gx[idx] = (T)p * std::pow(x[idx], (T)(p - 1.)) * gy[idx];
    }
  }
}

template <typename T>
void NormCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Norm<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void NormCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];
  const auto x_size = x->size();
  const auto y_size = y->size();

  // abs pow
  Variable out_abs_pow(x->shape()); // temporal buffer
  const auto data_x = x->get_data_pointer<Tcu>(this->ctx_);
  const auto data_out_abs_pow =
      out_abs_pow.cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_pow, x_size, data_x,
                                 data_out_abs_pow, this->p_);
  // sum
  Variable out_sum; // temporal buffer
  execute(this->sum_, {&out_abs_pow}, {&out_sum});
  // pow
  const auto data_out_sum = out_sum.get_data_pointer<Tcu>(this->ctx_);
  auto data_y = y->cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_pow, y_size, data_out_sum, data_y,
                                 1 / this->p_);
}

template <typename T>
void NormCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];
  const auto x_size = x->size();
  const auto y_size = y->size();

  // Forward
  // abs pow
  Variable out_abs_pow(x->shape()); // temporal buffer
  const auto data_x = x->get_data_pointer<Tcu>(this->ctx_);
  const auto data_out_abs_pow =
      out_abs_pow.cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_pow, x_size, data_x,
                                 data_out_abs_pow, this->p_);
  // sum
  Variable out_sum; // temporal buffer
  execute(this->sum_, {&out_abs_pow}, {&out_sum});
  // pow
  // (skipped since pow output is not need for backward calculation)

  // Backward
  // pow
  const auto data_out_sum = out_sum.get_data_pointer<Tcu>(this->ctx_);
  const auto grad_y = y->get_grad_pointer<Tcu>(this->ctx_);
  auto grad_out_sum = out_sum.cast_grad_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_pow_backward, y_size, data_out_sum,
                                 grad_y, grad_out_sum, 1 / this->p_);
  // sum
  nbla::backward(this->sum_, {&out_abs_pow}, {&out_sum}, propagate_down,
                 {false});
  // abs pow
  auto grad_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  const auto grad_out_abs_pow = out_abs_pow.get_grad_pointer<Tcu>(this->ctx_);
  const auto kernel = accum[0] ? kernel_abs_pow_backward<Tcu, true>
                               : kernel_abs_pow_backward<Tcu, false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, x_size, data_x, grad_out_abs_pow,
                                 grad_x, this->p_);
}
}
