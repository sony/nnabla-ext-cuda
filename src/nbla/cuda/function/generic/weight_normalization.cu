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

#include <algorithm>
#include <functional>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/sum.hpp>
#include <nbla/cuda/function/weight_normalization.hpp>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/cuda/utils/reduce.cuh>
#include <nbla/imperative.hpp>
#include <nbla/utils/nd_index.hpp>
#include <numeric>

namespace nbla {

int3 compute_folded_wstrides(Shape_t wshape, int dim) {
  Shape_t wshape_f;
  auto ndim = wshape.size();
  auto prod = [](Shape_t shape, int b, int e) {
    return std::accumulate(shape.begin() + b, shape.begin() + e, 1,
                           std::multiplies<int64_t>());
  };
  if (dim == 0) {
    wshape_f.push_back(1);
    wshape_f.push_back(wshape[0]);
    wshape_f.push_back(prod(wshape, 1, ndim));
  } else if (dim == ndim - 1) {
    wshape_f.push_back(prod(wshape, 0, ndim - 1));
    wshape_f.push_back(wshape[ndim - 1]);
    wshape_f.push_back(1);
  } else {
    wshape_f.push_back(prod(wshape, 0, dim));
    wshape_f.push_back(wshape[dim]);
    wshape_f.push_back(prod(wshape, dim + 1, ndim));
  }
  auto wstrides = ndi::strides(wshape_f);
  return to_int3(wstrides);
}

template <typename T>
__global__ void kernel_prod(const int size, const T *x, const T *y, T *tmp) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { tmp[idx] = x[idx] * y[idx]; }
}

template <typename T>
__global__ void
kernel_weight_normalization_forward(const int size, const T *w, const T *g,
                                    const T *sum_w2, T *w_WN,
                                    const int3 wstrides_f, float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto nd_idx = device_flat_to_4d(idx, wstrides_f);
    auto o = nd_idx.y;
    auto n = rsqrt(sum_w2[o] + eps);
    w_WN[idx] = g[o] * w[idx] * n;
  }
}

template <typename T, bool accum = false>
__global__ void kernel_weight_normalization_filter_backward(
    const int size, const T *w, const T *g, const T *sum_w2,
    const T *sum_dw_WN_x_w, const T *dw_WN, T *dw, const int3 wstrides_f,
    float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto nd_idx = device_flat_to_4d(idx, wstrides_f);
    auto o = nd_idx.y;
    auto s = sum_w2[o] + eps;
    auto n0 = rsqrt(s);
    auto n1 = n0 * n0 * n0;
    auto g_o = g[o];
    auto dw_i =
        (dw_WN[idx] * g_o * n0) - (sum_dw_WN_x_w[o] * g_o * n1 * w[idx]);
    dw[idx] = accum ? dw[idx] + dw_i : dw_i;
  }
}

template <typename T, bool accum = false>
__global__ void kernel_weight_normalization_scale_backward(
    const int size, const T *sum_w2, const T *sum_dw_WN_x_w, T *dg, float eps) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    auto sum_w2_eps = sum_w2[i] + eps;
    auto n0 = rsqrt(sum_w2_eps);
    auto dg_i = sum_dw_WN_x_w[i] * n0;
    dg[i] = accum ? dg[i] + dg_i : dg_i;
  }
}

template <typename T>
void WeightNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {
  WeightNormalization<T>::setup_impl(inputs, outputs);

  cuda_set_device(this->device_);

  auto ndim = inputs[0]->ndim();
  vector<int> axes;
  for (int a = 0; a < ndim; ++a) {
    if (a != this->dim_)
      axes.push_back(a);
  }
  f_sum_ = create_Sum(this->ctx_, axes, false);
}

template <typename T>
void WeightNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  cuda_set_device(this->device_);

  auto w0 = inputs[0];
  auto g0 = inputs[1];
  auto w_WN = outputs[0];
  auto wshape0 = w0->shape();

  auto size = w0->size();
  auto data_w = w0->get_data_pointer<Tcu>(this->ctx_);
  auto data_g = g0->get_data_pointer<Tcu>(this->ctx_);
  auto data_w_WN =
      w_WN->cast_data_and_get_pointer<Tcu>(this->ctx_); // temporal buf

  // sum(w^2)
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_prod, size, data_w, data_w, data_w_WN);
  Variable sum_w2;
  execute(f_sum_, {w_WN}, {&sum_w2});
  // weight normalization forward
  auto wstrides_f = compute_folded_wstrides(wshape0, this->dim_);
  auto data_sum_w2 = sum_w2.get_data_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_weight_normalization_forward, size,
                                 data_w, data_g, data_sum_w2, data_w_WN,
                                 wstrides_f, this->eps_);
}

template <typename T>
void WeightNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  // Compute:
  // dw_j = dw_WN_j * g * s^{-1/2} - (sum_i (dw_WN_i * w_i)) * g * s^{-3/2} *
  // w_j
  // dg = (sum_i (dw_WN_i * w_i)) * s^{-3/2}
  // where s = sum_i (w_i^2) + eps

  auto w0 = inputs[0];
  auto g0 = inputs[1];
  auto w_WN = outputs[0];
  auto wshape0 = w0->shape();

  auto size = w0->size();
  auto data_w = w0->get_data_pointer<Tcu>(this->ctx_);
  auto grad_w_WN = w_WN->get_grad_pointer<Tcu>(this->ctx_);
  auto data_w_WN =
      w_WN->cast_data_and_get_pointer<Tcu>(this->ctx_); // temporal buf

  // sum(w^2)
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_prod, size, data_w, data_w, data_w_WN);
  Variable sum_w2;
  execute(f_sum_, {w_WN}, {&sum_w2});

  // sum(dw_WN * w)
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_prod, size, data_w, grad_w_WN,
                                 data_w_WN);
  Variable sum_dw_WN_x_w;
  execute(f_sum_, {w_WN}, {&sum_dw_WN_x_w});

  auto data_sum_dw_WN_x_w = sum_dw_WN_x_w.get_data_pointer<Tcu>(this->ctx_);
  auto data_sum_w2 = sum_w2.get_data_pointer<Tcu>(this->ctx_);
  // wrt w
  if (propagate_down[0]) {
    auto size = w0->size();
    auto data_g = g0->get_data_pointer<Tcu>(this->ctx_);
    auto data_sum_dw_WN_x_w = sum_dw_WN_x_w.get_data_pointer<Tcu>(this->ctx_);
    auto grad_w = w0->cast_grad_and_get_pointer<Tcu>(this->ctx_);
    auto kernel = accum[0]
                      ? kernel_weight_normalization_filter_backward<Tcu, true>
                      : kernel_weight_normalization_filter_backward<Tcu, false>;
    auto wstrides_f = compute_folded_wstrides(wshape0, this->dim_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, data_w, data_g, data_sum_w2,
                                   data_sum_dw_WN_x_w, grad_w_WN, grad_w,
                                   wstrides_f, this->eps_);
  }

  // wrt g
  if (propagate_down[1]) {
    auto size = g0->size();
    auto grad_g = g0->cast_grad_and_get_pointer<Tcu>(this->ctx_);
    auto kernel = accum[1]
                      ? kernel_weight_normalization_scale_backward<Tcu, true>
                      : kernel_weight_normalization_scale_backward<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, data_sum_w2,
                                   data_sum_dw_WN_x_w, grad_g, this->eps_);
  }
}
}
