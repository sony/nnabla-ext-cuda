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

// -*- coding:utf-8 -*-
/*
 * Copyright (C) 2016 Sony Corporation
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Corporation;
 * the contents of this file is not to be disclosed to third parties, copied
 * or duplicated in any form, in whole or in part, without the prior written
 * permission of Sony Corporation.
 */

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/embed.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, typename T1>
__global__ void kernel_embed_forward(const int num, T1 *y, const T *x,
                                     const T1 *w, int stride0) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    const int i = idx / stride0;
    const int j = idx % stride0;
    y[idx] = w[x[i] * stride0 + j];
  }
}

template <typename T, typename T1>
__global__ void kernel_embed_backward_weight(const int num, T1 *dw, const T *x,
                                             const T1 *dy, int stride0) {
  // TODO: optimize
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    const int i = idx / stride0;
    const int j = idx % stride0;
    atomicAdd(dw + x[i] * stride0 + j, dy[i * stride0 + j]);
  }
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  Embed<T, T1>::setup_impl(inputs, outputs);
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T1 *w = inputs[1]->get_data_pointer<T1>(this->ctx_);
  T1 *y = outputs[0]->cast_data_and_get_pointer<T1>(this->ctx_);

  Size_t stride0 = inputs[1]->size(1);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_embed_forward,
                                 inputs[0]->size() * stride0, y, x, w, stride0);
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {

  NBLA_CHECK(!propagate_down[0], error_code::value,
             "Index array can not be propagated down.");
  if (!propagate_down[1]) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (!accum[1])
    inputs[1]->grad()->zero();
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T1 *dw = inputs[1]->cast_grad_and_get_pointer<T1>(this->ctx_);
  const T1 *dy = outputs[0]->get_grad_pointer<T1>(this->ctx_);

  Size_t stride0 = inputs[1]->size(1);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_embed_backward_weight,
                                 inputs[0]->size() * stride0, dw, x, dy,
                                 stride0);
}

// template instantiation
template class EmbedCuda<int, float>;
}
