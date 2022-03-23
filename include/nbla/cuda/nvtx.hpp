// Copyright 2021, 2022 Sony Corporation.
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

#ifndef __NBLA_NVTX_HPP__
#define __NBLA_NVTX_HPP__

#include "nbla/cuda/defs.hpp"
#include <string>

namespace nbla {
using std::string;

int dl_nvtx_init(void);
int dl_nvtx_finish(void);

NBLA_CUDA_API void nvtx_mark_A(string msg);
NBLA_CUDA_API void nvtx_range_push_A(string msg);
NBLA_CUDA_API void nvtx_range_push_with_C(string msg);
NBLA_CUDA_API void nvtx_range_pop();
}

#endif //__NBLA_NVTX_HPP__
