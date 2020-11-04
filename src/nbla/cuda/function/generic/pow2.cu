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

// pow2.cu

#include <nbla/cuda/function/pow2.hpp>
#include <nbla/cuda/function/utils/base_transform_binary.cuh>

#include <cmath>

namespace nbla {

NBLA_DEFINE_TRANSFORM_BINARY_CUDA(
    Pow2, std::pow(x0, x1),
    dy *x1 *std::pow(inplace ? std::pow(y, 1 / x1) : x0, x1 - (T)1),
    dy *std::log(inplace ? std::pow(y, 1 / x1) : x0) *
        std::pow(inplace ? std::pow(y, 1 / x1) : x0, x1));
}
