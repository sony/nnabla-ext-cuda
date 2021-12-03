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
#include <nbla/cuda/function/arange.hpp>
#include <nbla/cuda/function/linspace.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_make_sequence(const int num, T *output,
                                     const float start, const float stop,
                                     const double step) {
  NBLA_CUDA_KERNEL_LOOP(i, num) { output[i] = start + i * step; }
}

template <typename T>
__global__ void kernel_set_value(const Size_t size, T *output, const int idx,
                                 const float value) {
  output[idx] = value;
}

template <typename T>
void LinspaceCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);

  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  if (this->num_ > 1) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_make_sequence<Tcu>, this->num_, y,
                                   this->start_, this->stop_, this->step_);
  } else if (this->num_ == 1) {
    outputs[0]->data()->fill(this->start_);
  }
}
}
