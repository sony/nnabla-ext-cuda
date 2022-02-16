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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/norm_normalization.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/function/mul2.hpp>
#include <nbla/cuda/function/sum.hpp>

#include <cmath>

namespace nbla {

template <typename T>
__global__ void kernel_abs_pow(const int size, const T *x, T *y, float p) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = std::pow(abs(x[idx]), (T)p); }
}

template <typename T>
__global__ void kernel_add_pow(const int size, const T *x, T *y, float p,
                               float eps) {
  // optimization with replacing pow to rsqrt
  if (p == -0.5f) {
    NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = rsqrt(x[idx] + (T)eps); }
  } else {
    NBLA_CUDA_KERNEL_LOOP(idx, size) {
      y[idx] = std::pow(x[idx] + (T)eps, (T)p);
    }
  }
}

template <typename T>
__global__ void kernel_abs_pow_backward(const int size, const T *x, const T *gy,
                                        T *gx, const float p) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    // always accumulate
    if (x[idx] < 0) {
      gx[idx] = gx[idx] + -p * std::pow(abs(x[idx]), (T)(p - 1.)) * gy[idx];
    } else {
      gx[idx] = gx[idx] + p * std::pow(abs(x[idx]), (T)(p - 1.)) * gy[idx];
    }
  }
}

template <typename T>
__global__ void kernel_add_pow_backward(const int size, const T *x, const T *gy,
                                        T *gx, float p, float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    gx[idx] = (T)p * std::pow(x[idx] + (T)eps, (T)(p - 1.)) * gy[idx];
  }
}

template <typename T>
void NormNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  NormNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  // functions
  f_sum_ = create_Sum(this->ctx_, this->axes_, true /* keep_dims */);
  f_mul2_ = create_Mul2(this->ctx_, false);
}

template <typename T>
void NormNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];
  const int x_size = x->size();

  // abs pow
  const auto data_x = x->get_data_pointer<Tcu>(this->ctx_);
  auto data_y = y->cast_data_and_get_pointer<Tcu>(this->ctx_); // temporal buf
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_pow, x_size, data_x, data_y,
                                 this->p_);
  // sum
  Variable out_sum;
  execute(f_sum_, {y}, {&out_sum});
  // add pow
  Variable out_add_pow(out_sum.shape());
  const int out_sum_size = out_sum.size();
  const auto data_out_sum = out_sum.get_data_pointer<Tcu>(this->ctx_);
  auto data_out_add_pow =
      out_add_pow.cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_pow, out_sum_size, data_out_sum,
                                 data_out_add_pow, -1. / this->p_, this->eps_);
  // mul
  execute(f_mul2_, {x, &out_add_pow}, {y});
}

template <typename T>
void NormNormalizationCuda<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  const bool prop_down = propagate_down[0];
  if (!prop_down) {
    return;
  }
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];
  const int x_size = x->size();

  // forward
  // abs pow
  Variable out_abs_pow(x->shape());
  const auto data_x = x->get_data_pointer<Tcu>(this->ctx_);
  auto data_out_abs_pow =
      out_abs_pow.cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_pow, x_size, data_x,
                                 data_out_abs_pow, this->p_);
  // sum
  Variable out_sum;
  execute(f_sum_, {&out_abs_pow}, {&out_sum});
  // add pow
  Variable out_add_pow(out_sum.shape());
  const int out_sum_size = out_sum.size();
  const auto data_out_sum = out_sum.get_data_pointer<Tcu>(this->ctx_);
  auto data_out_add_pow =
      out_add_pow.cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_pow, out_sum_size, data_out_sum,
                                 data_out_add_pow, -1. / this->p_, this->eps_);
  // mul
  execute(f_mul2_, {x, &out_add_pow}, {y});

  // backward
  // mul
  nbla::backward(f_mul2_, {x, &out_add_pow}, {y}, {prop_down, prop_down},
                 {accum[0], false});
  // add pow
  const auto grad_out_add_pow = out_add_pow.get_grad_pointer<Tcu>(this->ctx_);
  auto grad_out_sum = out_sum.cast_grad_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_pow_backward, out_sum_size,
                                 data_out_sum, grad_out_add_pow, grad_out_sum,
                                 -1. / this->p_, this->eps_);
  // sum
  nbla::backward(f_sum_, {&out_abs_pow}, {&out_sum}, {prop_down}, {false});
  // abs pow
  const auto grad_out_abs_pow = out_abs_pow.get_grad_pointer<Tcu>(this->ctx_);
  auto grad_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_pow_backward, x_size, data_x,
                                 grad_out_abs_pow, grad_x, this->p_);
}
}
