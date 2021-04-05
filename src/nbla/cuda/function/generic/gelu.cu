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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/gelu.hpp>
#include <nbla/cuda/function/utils/base_transform_unary.cuh>

#include <cmath>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

namespace nbla {
NBLA_DEFINE_TRANSFORM_UNARY_CUDA(
    GELU, x / 2 * (1 + std::tanh((std::sqrt((T)(2 / M_PI)) *
                                  (x + (T)0.044715 * std::pow(x, (T)3))))),
    dy *(0.5 * (1 + std::tanh(std::sqrt((T)(2 / M_PI)) *
                              (x + (T)0.044715 * std::pow(x, T(3))))) +
         0.5 * x *
             (1 - std::pow(std::tanh(std::sqrt((T)(2 / M_PI)) *
                                     (x + (T)0.044715 * std::pow(x, T(3)))),
                           T(2))) *
             std::sqrt((T)(2 / M_PI)) * (1 + 0.134145 * std::pow(x, T(2)))),
    false, true);
}
