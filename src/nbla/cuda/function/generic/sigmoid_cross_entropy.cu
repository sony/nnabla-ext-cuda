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

// sigmoid_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/sigmoid_cross_entropy.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, typename Tl>
__global__ void kernel_sigmoid_cross_entropy_forward(const int size,
                                                     const T *x0, const Tl *x1,
                                                     T *y) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    y[s] = -(x0[s] * (x1[s] - (x0[s] >= 0)) -
             log(1 + exp(x0[s] - 2 * x0[s] * (x0[s] >= 0))));
  }
}

template <typename T, typename Tl, bool accum>
__global__ void kernel_sigmoid_cross_entropy_backward(const int size,
                                                      const T *dy, const T *x0,
                                                      const Tl *x1, T *dx0) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    dx0[s] = (accum ? dx0[s] : 0) + dy[s] * (1 / (1 + exp(-x0[s])) - x1[s]);
  }
}

template <typename T, typename Tl>
void SigmoidCrossEntropyCuda<T, Tl>::setup_impl(const Variables &inputs,
                                                const Variables &outputs) {
  SigmoidCrossEntropy<T, Tl>::setup_impl(inputs, outputs);
}

template <typename T, typename Tl>
void SigmoidCrossEntropyCuda<T, Tl>::forward_impl(const Variables &inputs,
                                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const Tl *x1 = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const Size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_sigmoid_cross_entropy_forward, size, x0,
                                 x1, y);
}

template <typename T, typename Tl>
void SigmoidCrossEntropyCuda<T, Tl>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[1], error_code::value,
             "Label can not be propagated down.");
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const Tl *x1 = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  const Size_t size = inputs[0]->size();
  if (propagate_down[0]) {
    T *dx0 = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_sigmoid_cross_entropy_backward<T, Tl, true>), size, dy, x0,
          x1, dx0);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_sigmoid_cross_entropy_backward<T, Tl, false>), size, dy, x0,
          x1, dx0);
    }
  }
}
}
