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

namespace nbla {

template <typename T> __device__ __inline__ T atomic_min(T *dst_adr, T val) {
  return atomicMin(dst_adr, val);
}

template <>
__device__ __inline__ unsigned long long int
atomic_min(unsigned long long int *dst_adr, unsigned long long int val) {
#if (__CUDA_ARCH__ >= 350)
  return atomicMin(dst_adr, val);
#else
  // `atomicMin()` for u64 is only supported for CC >= 3.5.
  // Here, we implement it using `atomicCAS()` based on sample codes in CUDA-C
  // Programming Guide by NVIDIA.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
  unsigned long long int old = *dst_adr, assumed;
  do {
    assumed = old;
    old = atomicCAS(dst_adr, assumed, std::min(assumed, val));
  } while (assumed != old);
  return old;
#endif
}
}
#endif
