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

// hard_sigmoid.cpp

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/hard_sigmoid.hpp>
#include <nbla/cuda/function/utils/base_transform_unary.cuh>

#include <cmath>

namespace nbla {
NBLA_DEFINE_TRANSFORM_UNARY_CUDA(HardSigmoid,
                                 x > (T)2.5 ? (T)1 : x < (T)-2.5
                                                         ? (T)0
                                                         : (T)0.2 * x + (T)0.5,
                                 x <= (T)2.5 && (T)-2.5 <= x ? (T)0.2 : (T)0);
}
