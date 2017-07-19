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

/** Cpu resources
*/
#ifndef __NBLA_CUDA_CUDA_HPP__
#define __NBLA_CUDA_CUDA_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda_memory.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla/exception.hpp>
#include <nbla/memory.hpp>
#include <nbla/singleton_manager.hpp>

#include <mutex>
#include <unordered_map>

namespace nbla {

using std::unordered_map;

/**
Singleton class for storing some handles or configs for CUDA Computation.
*/
class NBLA_CUDA_API Cuda {

public:
  ~Cuda();
  /** Get cuBLAS handle of a specified device */
  cublasHandle_t cublas_handle(int device = -1);

  /** Get cuRAND global generator **/
  curandGenerator_t curand_generator();

  void curand_set_seed(int seed);

  template <typename T> void curand_generate_uniform(T *r, int size);

  /** Available array class list used in CUDA Function implementations.
   */
  vector<string> array_classes() const;

  /** Set array class list.

      @note Dangerous to call. End users shouldn't call.
   */
  void _set_array_classes(const vector<string> &a);

  /** Register array class to available list by name.
   */
  void register_array_class(const string &name);

  /** Get a CudaMemoryCache instance.
   */
  MemoryCache<CudaMemory> &memcache();

protected:
  std::mutex mtx_cublas_;
  std::mutex mtx_curand_;
  unordered_map<int, cublasHandle_t>
      cublas_handles_; ///< cuBLAS handles for each device.
  unordered_map<int, curandGenerator_t> curand_generators_;
  vector<string> array_classes_;     ///< Available array classes
  MemoryCache<CudaMemory> memcache_; ///< CUDA memory cache.

private:
  friend SingletonManager;
  // Never called by users.
  Cuda();
  DISABLE_COPY_AND_ASSIGN(Cuda);
};
}
#endif
