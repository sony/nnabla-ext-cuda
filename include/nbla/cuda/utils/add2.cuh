
// Copyright 2021 Sony Group Corporation.
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

#ifndef __NBLA_CUDA_UTILS_ADD2_CUH__
#define __NBLA_CUDA_UTILS_ADD2_CUH__

#include <nbla/cuda/common.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_add2_forward(const int num, T *y, const T *x0,
                                    const T *x1) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x0[idx] + x1[idx]; }
}

template <typename T, bool accum>
__global__ void kernel_add2_backward(const int num, T *d, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    d[idx] = (accum ? d[idx] : (T)0) + dy[idx];
  }
}
} // namespace nbla

#endif
