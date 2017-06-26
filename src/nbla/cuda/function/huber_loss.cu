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

// huber_loss.cu

#include <nbla/cuda/function/huber_loss.hpp>
#include <nbla/cuda/function/utils/base_transform_binary.cuh>

#include <cmath>

namespace nbla {

NBLA_DEFINE_TRANSFORM_BINARY_CUDA_1(
    HuberLoss,
    (fabs(x0 - x1) < (T)a0) ? (fabs(x0 - x1) * fabs(x0 - x1))
                            : ((T)a0 *((T)2 * fabs(x0 - x1) - (T)a0)),
    (T)2 * dy * (fabs(x0 - x1) < (T)a0
                     ? (x0 - x1)
                     : ((x0 - x1) > (T)0 ? (T)1 : (T)-1) * (T)a0),
    -(T)2 * dy * (fabs(x0 - x1) < (T)a0
                      ? (x0 - x1)
                      : ((x0 - x1) > (T)0 ? (T)1 : (T)-1) * (T)a0),
    float);
// Template instantiation
template class HuberLossCuda<float>;
}
