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

#ifndef __NBLA_CUDA_UTILS_WARP_VOTE_CUH__
#define __NBLA_CUDA_UTILS_WARP_VOTE_CUH__

namespace nbla {
namespace warp {

__forceinline__ __device__ int all(int predicate) {
#if CUDA_VERSION >= 9000
  return __all_sync(0xfffffff, predicate);
#else  // !(CUDA_VERSION >= 9000)
  return __all(predicate);
#endif // CUDA_VERSION >= 9000
}

__forceinline__ __device__ int any(int predicate) {
#if CUDA_VERSION >= 9000
  return __any_sync(0xfffffff, predicate);
#else  // !(CUDA_VERSION >= 9000)
  return __any(predicate);
#endif // CUDA_VERSION >= 9000
}

__forceinline__ __device__ unsigned int ballot(int predicate) {
#if CUDA_VERSION >= 9000
  return __ballot_sync(0xfffffff, predicate);
#else  // !(CUDA_VERSION >= 9000)
  return __ballot(predicate);
#endif // CUDA_VERSION >= 9000
}

} // namespace warp
} // namespace nbla

#endif
