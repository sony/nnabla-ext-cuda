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

#ifndef __NBLA_CUDA_UTILS_ATOMIC_MIN_CUH__
#define __NBLA_CUDA_UTILS_ATOMIC_MIN_CUH__

#include <cuda.h>
#include <nbla/common.hpp>

namespace nbla {

template <typename T> __device__ __inline__ T atomic_min(T *dst_adr, T val) {
  return atomicMin(dst_adr, val);
}

template <>
__device__ __inline__ int64_t atomic_min(int64_t *dst_adr, int64_t val) {
  // `atomicMin()` for `signed long long int` does not exist.
  // Here, we implement it using `atomicCAS()` based on sample codes in CUDA-C
  // Programming Guide by NVIDIA.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  int64_t old = *dst_adr, assumed;
  while (val < old) {
    assumed = old;
    old = atomicCAS((unsigned long long *)dst_adr, (unsigned long long)assumed,
                    (unsigned long long)val);
    if (old == assumed) {
      break;
    }
  }
  return old;
}
}
#endif
