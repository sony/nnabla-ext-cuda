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
#include <nbla/cuda/function/add_n.hpp>
#include <nbla/cuda/utils/pointers.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_add_n_forward(const int size, const int num_inputs,
                                     const T **x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    T val = 0;
    for (int i = 0; i < num_inputs; i++) {
      val += x[i][idx];
    }
    y[idx] = val;
  }
}

template <typename T>
__global__ void
kernel_add_n_backward(const int num, const int num_inputs, T **dx, const T *dy,
                      const uint8_t *propdown, const uint8_t *accum) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    for (int i = 0; i < num_inputs; i++) {
      if (propdown[i])
        dx[i][idx] = (accum[i] ? dx[i][idx] : (T)0) + dy[idx];
    }
  }
}

template <typename T>
void AddNCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  AddN<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void AddNCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto xptrs = get_cuda_pointer_array<Tcu>(inputs, this->ctx_, [&](int i) {
    return inputs[i]->template get_data_pointer<Tcu>(this->ctx_);
  });
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_n_forward<Tcu>, inputs[0]->size(),
                                 inputs.size(),
                                 xptrs->template pointer<const Tcu *>(), y);
}

template <typename T>
void AddNCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto dxptrs = get_cuda_pointer_array<Tcu>(inputs, this->ctx_, [&](int i) {
    return inputs[i]->template cast_grad_and_get_pointer<Tcu>(this->ctx_,
                                                              !accum[i]);
  });
  auto propdown_array =
      create_ndarray_from_vector<bool, uint8_t>(propagate_down);
  auto accum_array = create_ndarray_from_vector<bool, uint8_t>(accum);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_add_n_backward<Tcu>), inputs[0]->size(), inputs.size(),
      dxptrs->template pointer<Tcu *>(), dy,
      propdown_array->cast(get_dtype<uint8_t>(), this->ctx_)
          ->template const_pointer<uint8_t>(),
      accum_array->cast(get_dtype<uint8_t>(), this->ctx_)
          ->template const_pointer<uint8_t>());
}
}
