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
#include <nbla/cuda/function/randn.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void RandnCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Randn<T>::setup_impl(inputs, outputs);
}

template <typename T>
void RandnCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(device_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  curand_generate_randn<T>(curand_generator_, this->mu_, this->sigma_, y,
                           outputs[0]->size());
}

template <typename T>
void RandnCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  // Pass
}
}
