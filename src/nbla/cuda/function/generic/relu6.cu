// Copyright 2019,2020,2021 Sony Corporation.
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

// relu6.cpp

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/relu6.hpp>
#include <nbla/cuda/function/utils/base_transform_unary.cuh>

#include <cmath>

namespace nbla {
NBLA_DEFINE_TRANSFORM_UNARY_CUDA(ReLU6, min(max(x, (T)0), (T)6),
                                 (T)0 < x && x < (T)6 ? dy : (T)0, false, true);
}
