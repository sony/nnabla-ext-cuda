// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/cuda/utils/random.hpp>

namespace nbla {
// Generate rand(low, high) values from output of curandGenerateUniform.
// curandGenerateUniform returns random values in (0, 1], but we need [low,
// high).
template <typename T>
__global__ void kernel_rand_post_process(int size, T *dev_ptr, T low, T high) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    dev_ptr[idx] = (T(1) - dev_ptr[idx]) * (high - low) + low;
  }
}

static __global__ void kernel_randint_post_process(int size, int *dev_ptr,
                                                   int low, int high) {
  float *f_ptr = reinterpret_cast<float *>(dev_ptr);
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    dev_ptr[idx] = (1.0f - f_ptr[idx]) * (high - low) + low;
  }
}

__global__ void kernel_curand_init(const int size, const int seed,
                                   const int offset, curandState *state) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    curand_init(seed, idx, offset, &state[idx]);
  }
}

template <>
void curand_generate_rand<float>(curandGenerator_t gen, float low, float high,
                                 float *dev_ptr, size_t size) {
  NBLA_CURAND_CHECK(curandGenerateUniform(gen, dev_ptr, size));
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_rand_post_process<float>), size,
                                 dev_ptr, low, high);
}

template <>
void curand_generate_rand<int>(curandGenerator_t gen, int low, int high,
                               int *dev_ptr, size_t size) {
  NBLA_CURAND_CHECK(
      curandGenerateUniform(gen, reinterpret_cast<float *>(dev_ptr), size));
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_randint_post_process, size, dev_ptr,
                                 low, high);
}

void curand_initialize(const int size, const int seed, const int offset,
                       curandState *state) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_curand_init, size, seed, offset, state);
}
}
