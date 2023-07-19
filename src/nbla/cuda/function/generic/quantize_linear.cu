// Copyright 2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/quantize_linear.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_saturate(int size, T *x, int min_range, int max_range) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    if (x[i] < min_range) {
      x[i] = min_range;
    } else if (x[i] > max_range) {
      x[i] = max_range;
    }
  }
}

template <typename T> __global__ void kernel_std_round(int size, T *x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { x[i] = round(x[i]); }
}

template <typename T>
__global__ void kernel_round_half_to_even(int size, T *x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    auto t = round(x[i]);
    if (abs(x[i] - t) == 0.5) {
      x[i] = round(x[i] * 0.5) * 2;
    } else {
      x[i] = t;
    }
  }
}

template <typename T>
void QuantizeLinearCuda<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  QuantizeLinear<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void QuantizeLinearCuda<T>::saturate(Variable *inp, int min_range,
                                     int max_range) {
  auto size = inp->size();
  Tcu *x = inp->cast_data_and_get_pointer<Tcu>(this->ctx_, false);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_saturate<Tcu>, size, x, min_range,
                                 max_range);
}

template <typename T>
void QuantizeLinearCuda<T>::round(Variable *inp, std::string round_mode) {
  auto size = inp->size();
  Tcu *x = inp->cast_data_and_get_pointer<Tcu>(this->ctx_, false);
  if (round_mode == "HALF_AWAY_FROM_ZERO") {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_std_round<Tcu>, size, x);
  } else if (round_mode == "HALF_TO_EVEN") {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_round_half_to_even, size, x);
  }
}
} // namespace nbla
