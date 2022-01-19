// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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

#include <nbla/cuda/function/binary_weight_convolution.hpp>

namespace nbla {

template <typename T>
void BinaryWeightConvolutionCuda<T>::setup_impl(const Variables &inputs,
                                                const Variables &outputs) {
  BinaryWeightConvolution<T>::setup_impl(inputs, outputs);
}

template <typename T>
void BinaryWeightConvolutionCuda<T>::forward_impl(const Variables &inputs,
                                                  const Variables &outputs) {
  BinaryWeightConvolution<T>::forward_impl(inputs, outputs);
}

template <typename T>
void BinaryWeightConvolutionCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  BinaryWeightConvolution<T>::backward_impl(inputs, outputs, propagate_down,
                                            accum);
}

} // namespace nbla
