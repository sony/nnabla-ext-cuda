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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/utils/random.hpp>

#include <random>

namespace nbla {
curandGenerator_t curand_create_generator(int seed) {
  // CURAND_RNG_PSEUDO_DEFAULT is CURAND_RNG_PSEUDO_XORWOW.
  curandGenerator_t gen;
  NBLA_CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  if (seed == -1) {
    seed = std::random_device()();
  }
  curand_set_seed(gen, seed);
  return gen;
}

void curand_destroy_generator(curandGenerator_t gen) {
  NBLA_CURAND_CHECK(curandDestroyGenerator(gen));
}

void curand_set_seed(curandGenerator_t gen, int seed) {
  NBLA_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
}

template <>
void curand_generate_randn<float>(curandGenerator_t gen, float mu, float sigma,
                                  float *dev_ptr, size_t size) {
  if (size % 2 != 0) {
    // Normal generator requires length with multiple of two.
    CudaCachedArray arr(
        size + 1, get_dtype<float>(),
        Context().set_device_id(std::to_string(cuda_get_device())));
    float *buff = arr.pointer<float>();
    NBLA_CURAND_CHECK(curandGenerateNormal(gen, buff, size + 1, mu, sigma));
    NBLA_CUDA_CHECK(cudaMemcpy(dev_ptr, buff, size * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    return;
  }
  NBLA_CURAND_CHECK(curandGenerateNormal(gen, dev_ptr, size, mu, sigma));
}
}
