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

#ifndef NBLA_CUDA_FUNCTION_ATAN2_HPP
#define NBLA_CUDA_FUNCTION_ATAN2_HPP

#include <nbla/cuda/function/utils/base_transform_binary.hpp>
#include <nbla/function/atan2.hpp>

namespace nbla {

/** @copydoc ATan2
*/
NBLA_DECLARE_TRANSFORM_BINARY_CUDA(ATan2);
}
#endif
