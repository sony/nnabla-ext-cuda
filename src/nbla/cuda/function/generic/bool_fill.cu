
// Copyright 2021,2022 Sony Group Corporation.
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
#include <nbla/cuda/function/bool_fill.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

namespace nbla {

namespace bool_fill_cuda {

template <typename T>
__global__ void kernel_bool_fill_forward(const int N, T *output, const T *data,
                                         const T *mask, const T value) {
  NBLA_CUDA_KERNEL_LOOP(i, N) {
    output[i] = (mask[i] != 0) ? T(value) : data[i];
  }
}

template <typename T, bool accum = false>
__global__ void kernel_bool_fill_data_backward(const int N, T *g_data,
                                               const T *g_output,
                                               const T *mask) {
  NBLA_CUDA_KERNEL_LOOP(i, N) {
    auto mask_i = T(mask[i] != T(0));
    g_data[i] = accum ? g_data[i] + g_output[i] * (T(1) - mask_i)
                      : g_output[i] * (T(1) - mask_i);
  }
}
}

template <typename T>
void BoolFillCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  BoolFill<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void BoolFillCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);

  auto data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto output = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);

  auto N = inputs[0]->size();
  if (this->broadcast_func_ != nullptr) {
    Variable bmask;
    nbla::execute(this->broadcast_func_, {inputs[1]}, {&bmask});
    mask = bmask.get_data_pointer<Tcu>(this->ctx_);
    auto kernel = bool_fill_cuda::kernel_bool_fill_forward<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, output, data, mask,
                                   T(this->value_));
  } else {
    auto kernel = bool_fill_cuda::kernel_bool_fill_forward<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, output, data, mask,
                                   T(this->value_));
  }
}

template <typename T>
void BoolFillCuda<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  auto data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  auto g_data =
      inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  auto g_output = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto N = inputs[0]->size();

  if (propagate_down[0]) {
    if (this->broadcast_func_ != nullptr) {
      Variable bmask;
      nbla::execute(this->broadcast_func_, {inputs[1]}, {&bmask});
      mask = bmask.get_data_pointer<Tcu>(this->ctx_);
      auto kernel =
          accum[0] ? bool_fill_cuda::kernel_bool_fill_data_backward<Tcu, true>
                   : bool_fill_cuda::kernel_bool_fill_data_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, g_data, g_output, mask);
    } else {
      auto kernel =
          accum[0] ? bool_fill_cuda::kernel_bool_fill_data_backward<Tcu, true>
                   : bool_fill_cuda::kernel_bool_fill_data_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, N, g_data, g_output, mask);
    }
  }
}
}
