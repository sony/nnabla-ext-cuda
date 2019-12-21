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
#include <nbla/cuda/function/adaptive_separable_convolution.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

// TODO: Remove these #includes. Only for debug.
#include <iostream>
#include <typeinfo>

namespace nbla {

__device__ int4 idx_to_4d(int idx, int4 strides) {
  auto b = idx / strides.x;
  idx -= b * strides.x;
  auto c = idx / strides.y;
  idx -= c * strides.y;
  auto h = idx / strides.z;
  idx -= h * strides.z;
  auto w = idx;
  return make_int4(b, c, h, w);
}

template <typename T>
__global__ void kernel_adaptive_separable_convolution_forward(
    const int osize, T *y, const T *x, const T *kv, const T *kh,
    const int4 y_strides, const int4 x_strides, const int4 kv_strides,
    const int4 kh_strides, const int kv_filters, const int kh_filters) {
  NBLA_CUDA_KERNEL_LOOP(idx, osize) {
    auto bchw = idx_to_4d(idx, y_strides);
    auto b = bchw.x;
    auto c = bchw.y;
    auto h = bchw.z;
    auto w = bchw.w;
    auto x_bc = x + (x_strides.x * b + x_strides.y * c);
    auto kv_b = kv + (kv_strides.x * b);
    auto kh_b = kh + (kh_strides.x * b);

    // sum_{i, j} K_h(i, h, w) * K_v(j, h, w) * I(c, h+j, w+i)
    T val = T(0.0);
    for (int j = 0; j < kv_filters; ++j) {
      for (int i = 0; i < kh_filters; ++i) {
        auto kval = kv_b[kv_strides.y * j + kv_strides.z * h + w] *
                    kh_b[kh_strides.y * i + kh_strides.z * h + w];
        auto pval = x_bc[x_strides.z * (h + j) + (w + i)];
        val += kval * pval;
      }
    }
    y[idx] = val;
  }
}

template <typename T>
__global__ void kernel_adaptive_separable_convolution_input_backward(
    const int osize, const T *g_y, T *g_x, const T *kv, const T *kh,
    const int4 y_strides, const int4 x_strides, const int4 kv_strides,
    const int4 kh_strides, const int kv_filters, const int kh_filters) {
  NBLA_CUDA_KERNEL_LOOP(idx, osize) {
    auto bchw = idx_to_4d(idx, y_strides);
    auto b = bchw.x;
    auto c = bchw.y;
    auto h = bchw.z;
    auto w = bchw.w;
    auto g_y_bchw = g_y[idx];
    auto kv_b = kv + (kv_strides.x * b);
    auto kh_b = kh + (kh_strides.x * b);
    auto g_x_bc = g_x + (x_strides.x * b + x_strides.y * c);

    // g_y(c, h, w) * Kv(j, h, w) * Kh(i, h, w)
    for (int j = 0; j < kv_filters; ++j) {
      for (int i = 0; i < kh_filters; ++i) {
        auto kv_val = kv_b[kv_strides.y * j + kv_strides.z * h + w];
        auto kh_val = kh_b[kh_strides.y * i + kh_strides.z * h + w];
        auto val = g_y_bchw * kv_val * kh_val;
        atomic_add(&g_x_bc[x_strides.z * (h + j) + (w + i)], val);
      }
    }
  }
}

template <typename T, bool accum>
__global__ void kernel_adaptive_separable_convolution_vertical_weight_backward(
    const int kv_size, const T *g_y, T *g_kv, const T *x, const T *kh,
    const int4 y_strides, const int4 x_strides, const int4 kv_strides,
    const int4 kh_strides, const int imaps, const int kh_filters,
    const int2 o_sshape) {
  NBLA_CUDA_KERNEL_LOOP(idx, kv_size) {
    auto bchw = idx_to_4d(idx, kv_strides);
    auto b = bchw.x;
    auto j = bchw.y;
    auto h = bchw.z;
    auto w = bchw.w;

    auto oH = o_sshape.x;
    auto oW = o_sshape.y;
    if (h >= oH || w >= oW)
      return;

    // sum_{c} (sum_{i} K_h(i, h, w) * I(c, h+j, w+i)) * g_y(c, h, w))
    auto kh_b = kh + kh_strides.x * b;
    auto x_b = x + x_strides.x * b;
    auto g_y_b = g_y + y_strides.x * b;
    auto osum = T(0.0);
    for (int c = 0; c < imaps; ++c) {
      auto isum = T(0.0);
      for (int i = 0; i < kh_filters; ++i) {
        auto kval = kh_b[kh_strides.y * i + kh_strides.z * h + w];
        auto pval = x_b[x_strides.y * c + x_strides.z * (h + j) + (w + i)];
        isum += kval * pval;
      }
      osum += g_y_b[y_strides.y * c + y_strides.z * h + w] * isum;
    }
    g_kv[idx] = accum ? g_kv[idx] + osum : osum;
  }
}

template <typename T, bool accum>
__global__ void
kernel_adaptive_separable_convolution_horizontal_weight_backward(
    const int kh_size, const T *g_y, T *g_kh, const T *x, const T *kv,
    const int4 y_strides, const int4 x_strides, const int4 kv_strides,
    const int4 kh_strides, const int imaps, const int kv_filters,
    const int2 o_sshape) {
  NBLA_CUDA_KERNEL_LOOP(idx, kh_size) {
    auto bchw = idx_to_4d(idx, kh_strides);
    auto b = bchw.x;
    auto i = bchw.y;
    auto h = bchw.z;
    auto w = bchw.w;

    auto oH = o_sshape.x;
    auto oW = o_sshape.y;
    if (h >= oH || w >= oW)
      return;

    // sum_{c} (sum_{j} K_v(j, h, w) * I(c, h+j, w+i)) * g_y(c, h, w))
    auto kv_b = kv + kv_strides.x * b;
    auto x_b = x + x_strides.x * b;
    auto g_y_b = g_y + y_strides.x * b;
    auto osum = T(0.0);
    for (int c = 0; c < imaps; ++c) {
      auto isum = T(0.0);
      for (int j = 0; j < kv_filters; ++j) {
        auto kval = kv_b[kv_strides.y * j + kv_strides.z * h + w];
        auto pval = x_b[x_strides.y * c + x_strides.z * (h + j) + (w + i)];
        isum += kval * pval;
      }
      osum += g_y_b[y_strides.y * c + y_strides.z * h + w] * isum;
    }
    g_kh[idx] = accum ? g_kh[idx] + osum : osum;
  }
}

template <typename T>
void AdaptiveSeparableConvolutionCuda<T>::setup_impl(const Variables &inputs,
                                                     const Variables &outputs) {
  AdaptiveSeparableConvolution<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void AdaptiveSeparableConvolutionCuda<T>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  // TODO: it could be optimized
  cuda_set_device(this->device_);

  auto osize = outputs[0]->size();
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *kv = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *kh = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  auto y_strides =
      make_int4(outputs[0]->strides()[0], outputs[0]->strides()[1],
                outputs[0]->strides()[2], outputs[0]->strides()[3]);
  auto x_strides = make_int4(inputs[0]->strides()[0], inputs[0]->strides()[1],
                             inputs[0]->strides()[2], inputs[0]->strides()[3]);
  auto kv_strides = make_int4(inputs[1]->strides()[0], inputs[1]->strides()[1],
                              inputs[1]->strides()[2], inputs[1]->strides()[3]);
  auto kh_strides = make_int4(inputs[2]->strides()[0], inputs[2]->strides()[1],
                              inputs[2]->strides()[2], inputs[2]->strides()[3]);
  auto kv_filters = inputs[1]->shape()[1];
  auto kh_filters = inputs[2]->shape()[1];

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_adaptive_separable_convolution_forward<Tcu>, osize, y, x, kv, kh,
      y_strides, x_strides, kv_strides, kh_strides, kv_filters, kh_filters);
}

template <typename T>
void AdaptiveSeparableConvolutionCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  // TODO: it could be optimized
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  Tcu *g_x{nullptr};
  Tcu *g_kv{nullptr};
  Tcu *g_kh{nullptr};
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *kv = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *kh = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  auto osize = outputs[0]->size();
  auto kv_size = inputs[1]->size();
  auto kh_size = inputs[2]->size();
  auto y_strides =
      make_int4(outputs[0]->strides()[0], outputs[0]->strides()[1],
                outputs[0]->strides()[2], outputs[0]->strides()[3]);
  auto x_strides = make_int4(inputs[0]->strides()[0], inputs[0]->strides()[1],
                             inputs[0]->strides()[2], inputs[0]->strides()[3]);
  auto kv_strides = make_int4(inputs[1]->strides()[0], inputs[1]->strides()[1],
                              inputs[1]->strides()[2], inputs[1]->strides()[3]);
  auto kh_strides = make_int4(inputs[2]->strides()[0], inputs[2]->strides()[1],
                              inputs[2]->strides()[2], inputs[2]->strides()[3]);
  const auto kv_filters = inputs[1]->shape()[1];
  const auto kh_filters = inputs[2]->shape()[1];
  const auto imaps = inputs[0]->shape()[1];
  auto o_sshape = make_int2(outputs[0]->shape()[2], outputs[0]->shape()[3]);

  if (propagate_down[0]) {
    g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        kernel_adaptive_separable_convolution_input_backward, osize, g_y, g_x,
        kv, kh, y_strides, x_strides, kv_strides, kh_strides, kv_filters,
        kh_filters);
  }
  if (propagate_down[1]) {
    g_kv = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
    auto kernel =
        accum[1]
            ? (kernel_adaptive_separable_convolution_vertical_weight_backward<
                  Tcu, true>)
            : (kernel_adaptive_separable_convolution_vertical_weight_backward<
                  Tcu, false>);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, kv_size, g_y, g_kv, x, kh, y_strides,
                                   x_strides, kv_strides, kh_strides, imaps,
                                   kh_filters, o_sshape);
  }
  if (propagate_down[2]) {
    g_kh = inputs[2]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[2]);
    auto kernel =
        accum[2]
            ? (kernel_adaptive_separable_convolution_horizontal_weight_backward<
                  Tcu, true>)
            : (kernel_adaptive_separable_convolution_horizontal_weight_backward<
                  Tcu, false>);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, kh_size, g_y, g_kh, x, kv, y_strides,
                                   x_strides, kv_strides, kh_strides, imaps,
                                   kv_filters, o_sshape);
  }
}
}
