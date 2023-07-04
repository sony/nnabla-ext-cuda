// Copyright 2018,2019,2020,2021 Sony Corporation.
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

// add2.cu

#include <algorithm>
#include <cmath>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/add2.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/add2.cuh>

namespace nbla {

template <class T>
void Add2Cuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x0 = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *x1 = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add2_forward, size, y, x0, x1);
}

template <class T>
void Add2Cuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1]))
    return;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  size_t size = inputs[0]->size();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      Tc *dx = inputs[i]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[i]);
      if (dx != dy) {
        if (accum[i]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<Tc, true>), size,
                                         dx, dy);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<Tc, false>),
                                         size, dx, dy);
        }
      }
    }
  }
}
} // namespace nbla
