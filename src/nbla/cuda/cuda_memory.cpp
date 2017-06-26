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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda_memory.hpp>
#include <nbla/garbage_collector.hpp>
#include <nbla/memory-internal.hpp>

#include <iostream>
#include <memory>
#include <vector>

namespace nbla {

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;

/////////////////////////////
// CUDA Memory implementation
/////////////////////////////
CudaMemory::CudaMemory(Size_t bytes, const string &device)
    : Memory(bytes, device), device_num_(std::stoi(device)) {}
CudaMemory::~CudaMemory() {
  if (!ptr_)
    return;
  cuda_set_device(device_num_);
  cudaError_t err = cudaFree(ptr_);
  if (err != cudaSuccess) {
    if (err == cudaErrorInvalidDevicePointer) {
      // Workaround for invalid device pointer error by other cause.
      std::cout << "[Warn] CUDA throws `invalid device pointer` error. Trying "
                   "to exit "
                   "with success state as a workaround since we assume you are "
                   "seeing this error when exiting Python, and we presume the "
                   "GC is collecting a CUDA device pointer after some "
                   "finalization processes of CUDA which free all device "
                   "pointer left are called. For now, we consider this problem "
                   "as independent to NNabla. We observe this especially "
                   "when we use NNabla together with Theano. There might be "
                   "some conflicts among them. More investigation would be "
                   "required."
                << std::endl;
      ::exit(0); // Exit with success.
    } else {
      NBLA_ERROR(error_code::target_specific, "(%s) failed with \"%s\".", err,
                 cudaGetErrorString(err));
    }
  }
}

bool CudaMemory::allocate() {
  if (ptr_)
    return true;
  cuda_set_device(device_num_);
  try {
    NBLA_CUDA_CHECK(cudaMalloc(&ptr_, size_));
  } catch (...) {
    // Garbage collection and retry to allocate.
    SingletonManager::get<GarbageCollector>()->collect();
    try {
      cuda_set_device(device_num_);
      NBLA_CUDA_CHECK(cudaMalloc(&ptr_, size_));
    } catch (...) {
      return false;
    }
  }
  return true;
}

template class MemoryCache<CudaMemory>;
} // End of namespace nbla
