// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

// relu.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/relu.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/relu.cuh>

namespace nbla {

//=============================================================================
// forward_impl and backward_impl
//=============================================================================
template <class T>
void ReLUCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  const Size_t size2 = interpret_size<T>(size);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T(kernel_relu_forward, size2, size, y, x);
}

template <class T>
void ReLUCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *y = outputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Size_t size = inputs[0]->size();
  const Size_t size2 = interpret_size<T>(size);

  if (dx != dy && accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T((kernel_relu_backward<true>), size2,
                                          size, dx, y, dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T((kernel_relu_backward<false>), size2,
                                          size, dx, y, dy);
  }
}
} // namespace nbla
