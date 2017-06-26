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

#include <nbla/cuda/cuda.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {

Cuda::Cuda() {
  // CURAND_RNG_PSEUDO_DEFAULT is CURAND_RNG_PSEUDO_XORWOW.
  NBLA_CURAND_CHECK(
      curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
}

Cuda::~Cuda() {
  for (auto handle : this->cublas_handles_) {
    NBLA_CUBLAS_CHECK(cublasDestroy(handle.second));
  }
  NBLA_CURAND_CHECK(curandDestroyGenerator(curand_generator_));
}

cublasHandle_t Cuda::cublas_handle(int device) {
  if (device < 0) {
    NBLA_CUDA_CHECK(cudaGetDevice(&device));
  }
  if (this->cublas_handles_.count(device) == 0) {
    cublasHandle_t handle;
    NBLA_CUBLAS_CHECK(cublasCreate(&handle));
    this->cublas_handles_[device] = handle;
  }
  return this->cublas_handles_[device];
}

curandGenerator_t Cuda::curand_generator() { return this->curand_generator_; }

template <> void Cuda::curand_set_seed<float>(float seed) {
  NBLA_CURAND_CHECK(
      curandSetPseudoRandomGeneratorSeed(curand_generator_, seed));
}

template <> void Cuda::curand_generate_uniform<float>(float *r, int size) {
  NBLA_CURAND_CHECK(curandGenerateUniform(curand_generator_, r, size));
}

vector<string> Cuda::array_classes() const { return array_classes_; }

void Cuda::_set_array_classes(const vector<string> &a) { array_classes_ = a; }

void Cuda::register_array_class(const string &name) {
  array_classes_.push_back(name);
}

MemoryCache<CudaMemory> &Cuda::memcache() { return memcache_; }

NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, Cuda);
}
