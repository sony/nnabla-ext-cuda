// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_CUDA_UTILS_ATOMIC_ADD_CUH__
#define __NBLA_CUDA_UTILS_ATOMIC_ADD_CUH__

#include <cuda.h>
#include <cuda_fp16.h>
#include <nbla/cuda/half.hpp>

namespace nbla {

template <typename T>
__device__ __inline__ T atomic_add(T *dst_adr, T add_val) {
  return atomicAdd(dst_adr, add_val);
}

template <>
__device__ __inline__ HalfCuda atomic_add(HalfCuda *dst_adr, HalfCuda add_val) {
#if (CUDA_VERSION >= 10000) && (__CUDA_ARCH__ >= 700)
  return atomicAdd(&(dst_adr->h), add_val.h);
#else
  unsigned int *address;
  unsigned int old_int;
  unsigned int new_int;
  unsigned int compare;
  HalfCuda old_val;
  HalfCuda new_val;

  if ((size_t(dst_adr) & 2) == 0) {
    address = (unsigned int *)dst_adr;
    old_int = *address;
    do {
      old_val = __ushort_as_half(old_int & 0xffff);
      new_val = old_val + add_val;
      new_int = (old_int & 0xffff0000) | __half_as_ushort(new_val.h);
      compare = old_int;
      old_int = atomicCAS(address, compare, new_int);
    } while (old_int != compare);
  } else {
    address = (unsigned int *)(dst_adr - 1);
    old_int = *address;
    do {
      old_val = __ushort_as_half(old_int >> 16);
      new_val = old_val + add_val;
      new_int = (old_int & 0xffff) | (__half_as_ushort(new_val.h) << 16);
      compare = old_int;
      old_int = atomicCAS(address, compare, new_int);
    } while (old_int != compare);
  }
  return old_val;
#endif
}

} // namespace nbla
#endif
