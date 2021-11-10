// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/cumprod.hpp>
#include <nbla/variable.hpp>

#include "kernel/cumprod.cu"

namespace nbla {

template <typename T>
void CumProdCuda<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  CumProd<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  scan_setup_(inputs[0]->shape(), this->axis_, this->exclusive_,
              this->reverse_);
}

template <typename T>
void CumProdCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  device_cumprod(this->ctx_, x, y, scan_setup_, false /* accum */);
}

template <typename T>
void CumProdCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);

  device_cumprod_backward(this->ctx_, g_y, x, g_x, scan_setup_, accum[0]);
}
}