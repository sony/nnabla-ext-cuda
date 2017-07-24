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
#include <nbla/cuda/utils/random.hpp>
#include <nbla/singleton_manager-internal.hpp>

namespace nbla {

Cuda::Cuda() {}

Cuda::~Cuda() {
  for (auto handle : this->cublas_handles_) {
    NBLA_CUBLAS_CHECK(cublasDestroy(handle.second));
  }
  for (auto gen : this->curand_generators_) {
    curand_destroy_generator(gen.second);
  }
}

cublasHandle_t Cuda::cublas_handle(int device) {
  if (device < 0) {
    device = cuda_get_device();
  }
  std::lock_guard<decltype(mtx_cublas_)> lock(mtx_cublas_);
  auto it = this->cublas_handles_.find(device);
  // Create a new one
  if (it == this->cublas_handles_.end()) {
    cublasHandle_t handle;
    NBLA_CUBLAS_CHECK(cublasCreate(&handle));
    this->cublas_handles_.insert({device, handle});
    return handle;
  }
  return it->second;
}

curandGenerator_t Cuda::curand_generator() {
  // Get current device
  int device = cuda_get_device();
  std::lock_guard<decltype(mtx_curand_)> lock(mtx_curand_);
  // Find device rng
  auto it = this->curand_generators_.find(device);
  // Create a new one
  if (it == this->curand_generators_.end()) {
    curandGenerator_t gen = curand_create_generator();
    this->curand_generators_.insert({device, gen});
    return gen;
  }
  return it->second;
}

void Cuda::curand_set_seed(int seed) {
  ::nbla::curand_set_seed(curand_generator(), seed);
}

template <> void Cuda::curand_generate_uniform<float>(float *r, int size) {
  ::nbla::curand_generate_rand(curand_generator(), float(0), float(1), r, size);
}

vector<string> Cuda::array_classes() const { return array_classes_; }

void Cuda::_set_array_classes(const vector<string> &a) { array_classes_ = a; }

void Cuda::register_array_class(const string &name) {
  array_classes_.push_back(name);
}

MemoryCache<CudaMemory> &Cuda::memcache() { return memcache_; }

NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, Cuda);
}
