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

#ifndef __NBLA_CUDA_UTILS_RANDOM_CUH__
#define __NBLA_CUDA_UTILS_RANDOM_CUH__

#include <curand_kernel.h>

// curand_uniform returns random values in (0, 1], but we need [low,
// high).

__forceinline__ __device__ float curand_uniform(curandState *local_state,
                                                float low, float high) {
  return (high - low) * (1.0 - curand_uniform(local_state)) + low;
}

__forceinline__ __device__ float curand_normal(curandState *local_state,
                                               float mean, float stddev) {
  return mean + stddev * curand_normal(local_state);
}

#endif