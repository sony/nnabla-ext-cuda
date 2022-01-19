// Copyright 2020,2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/mish.hpp>
#include <nbla/cuda/function/utils/base_transform_unary.cuh>

#include <cmath>

namespace nbla {

NBLA_DEFINE_TRANSFORM_UNARY_CUDA(
    Mish, x *std::tanh(std::log(std::exp(x) + (T)1)),
    (dy * std::exp(x) *
     ((T)4 * (x + (T)1) + (T)4 * (std::exp((T)2 * x)) + std::exp((T)3 * x) +
      std::exp(x) * ((T)4 * x + (T)6))) /
        std::pow((T)2 * std::exp(x) + std::exp((T)2 * x) + (T)2, (T)2),
    false, true);
}
