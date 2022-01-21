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
#include <nbla/cuda/function/arange.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void make_sequence(const Size_t size, T *dst, const float start,
                              const float step) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dst[i] = start + i * step; }
}

template <typename T>
void ArangeCuda<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  if (outputs[0]->size() > 0) {
    cuda_set_device(this->device_);
    auto y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(make_sequence<Tcu>, outputs[0]->size(), y,
                                   this->start_, this->step_);
  }
}

} // namespace nbla
