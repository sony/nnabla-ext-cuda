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


#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/einsum.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void EinsumCuda<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  Einsum<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void EinsumCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu* x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu* y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
   Einsum<T>::forward_impl(inputs, outputs);
}


template <typename T>
void EinsumCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  cuda_set_device(this->device_);
  Einsum<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
