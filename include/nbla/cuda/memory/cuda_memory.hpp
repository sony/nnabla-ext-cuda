// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#ifndef __NBLA_CUDA_MEMORY_HPP__
#define __NBLA_CUDA_MEMORY_HPP__

#include <memory>
#include <unordered_map>
#include <vector>

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/memory/cpu_memory.hpp>
#include <nbla/memory/memory.hpp>

namespace nbla {

/** CUDA memory implementation.

    A CUDA device memory block allocated by cudaMalloc function is managed by
    this.

    The device passed to constructor is a device id as as string such as "0" and
    "1".

    \ingroup MemoryImplGrp
*/
class NBLA_CUDA_API CudaMemory : public Memory {
private:
  CudaMemory(size_t bytes, const string &device, void *ptr);

protected:
  int device_num_;

public:
  CudaMemory(size_t bytes, const string &device);
  ~CudaMemory();
  bool alloc_impl() override;
  shared_ptr<Memory> divide_impl(size_t second_start) override;
  void merge_next_impl(Memory *from) override;
  void merge_prev_impl(Memory *from) override;
};

/** CUDA memory implementation using unified memory
*/
class NBLA_CUDA_API CudaUnifiedMemory : public CudaMemory {
private:
  CudaUnifiedMemory(size_t bytes, const string &device, void *ptr);

public:
  CudaUnifiedMemory(size_t bytes, const string &device);

  bool alloc_impl() override;
  shared_ptr<Memory> divide_impl(size_t second_start) override;
};

/** Pinned host memory implementation
*/
class NBLA_CUDA_API CudaPinnedHostMemory : public CpuMemory {
  CudaPinnedHostMemory(size_t bytes, const string &device_id, void *ptr);

public:
  CudaPinnedHostMemory(size_t bytes, const string &device_id);
  ~CudaPinnedHostMemory();

protected:
  bool alloc_impl() override;
  shared_ptr<Memory> divide_impl(size_t second_start) override;
};
}
#endif
