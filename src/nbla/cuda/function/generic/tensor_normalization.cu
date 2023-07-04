// Copyright 2021 Sony Corporation.
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
#include <nbla/cuda/function/tensor_normalization.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void TensorNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {
  TensorNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void TensorNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  cuda_set_device(this->device_);
  TensorNormalization<T>::forward_impl(inputs, outputs);
}

template <typename T>
void TensorNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);
  TensorNormalization<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
} // namespace nbla
