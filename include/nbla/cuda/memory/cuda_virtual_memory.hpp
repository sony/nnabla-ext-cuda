// Copyright 2021 Sony Corporation.
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

#pragma once

#include <memory>

#include <nbla/array.hpp>
#include <unordered_map>
#include <vector>

#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/cuda/event.hpp>
#include <nbla/memory/memory.hpp>

// Todo: avoid including cudnn.h in cuda package.
#include <cudnn.h>

#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000

namespace nbla {

using std::make_shared;
using std::pair;

// ----------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------
void set_device_primary_ctx(int device_id);

CUmemAllocationProp &get_mem_allocation_prop(int device_id);

CUmemAccessDesc get_mem_access_desc(int device_id);

size_t get_allocation_granularity(int device_id);

size_t round_up_by_chunk(size_t x, int device_id);

// ----------------------------------------------------------------------
// CudaPhysicalMemory
// ----------------------------------------------------------------------
class NBLA_CUDA_API CudaPhysicalMemory : public PhysicalMemory {
private:
  CUmemGenericAllocationHandle handle_;

public:
  CudaPhysicalMemory(size_t bytes, const string &device_id)
      : PhysicalMemory(bytes, device_id), handle_(){};

  ~CudaPhysicalMemory();

  size_t alloc() override;

  inline CUmemGenericAllocationHandle &get_handle() { return handle_; };
};

// ----------------------------------------------------------------------
// CudaVirtualMemory
// ----------------------------------------------------------------------
class NBLA_CUDA_API CudaVirtualMemory : public Memory {
  CUdeviceptr dev_ptr_;
  CudaEvent event_;

  vector<pair<CUdeviceptr, size_t>> va_ranges_;

  void free_virtual_address();

public:
  // disable copy & move
  CudaVirtualMemory(const CudaVirtualMemory &) = delete;
  CudaVirtualMemory(CudaVirtualMemory &&) = delete;
  CudaVirtualMemory &operator=(const CudaVirtualMemory &) = delete;
  CudaVirtualMemory &operator=(CudaVirtualMemory &&) = delete;

  // ctor
  CudaVirtualMemory(size_t bytes, const string &device_id,
                    VecPhysicalMemoryPtr p_memories);

  // dtor
  ~CudaVirtualMemory();

  // make CUdeviceptr void_ptr
  inline void *get_pointer() { return (void *)(dev_ptr_); };

protected:
  void bind_impl() override;

  void unbind_impl() override;

  bool grow_impl(VecPhysicalMemoryPtr &p_mems) override;

  DeviceMemoryState get_device_memory_state() override;

  void lock_device_memory() override;

  // disable methods not needed in this class.
  bool alloc_impl() override {
    // call bind_impl()?
    NBLA_ERROR(error_code::memory,
               "CudaVirtualMemory doesn't have alloc_impl().");
  }

  shared_ptr<Memory> divide_impl(size_t second_start) override {
    NBLA_ERROR(error_code::memory,
               "CudaVirtualMemory doesn't have divide_impl().");
  }

  void merge_next_impl(Memory *from) override{/* do nothing */};
  void merge_prev_impl(Memory *from) override{/* do nothing */};
};
}

#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
