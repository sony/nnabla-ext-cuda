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
#include <nbla/cuda/memory/cuda_memory.hpp>

#include <memory>

#if 0
#include <cstdio>
#define DEBUG_LOG(...) printf(__VA_ARGS__);
#else
#define DEBUG_LOG(...)
#endif

namespace nbla {
using std::make_shared;
// ----------------------------------------------------------------------
// CudaMemory implementation
// ----------------------------------------------------------------------
CudaMemory::CudaMemory(size_t bytes, const string &device_id)
    : Memory(bytes, device_id), device_num_(std::stoi(device_id)) {}
CudaMemory::CudaMemory(size_t bytes, const string &device_id, void *ptr)
    : CudaMemory(bytes, device_id) {
  ptr_ = ptr;
}

CudaMemory::~CudaMemory() {
  if (!ptr_) {
    return;
  }
  NBLA_FORCE_ASSERT(!prev(),
                    "Trying to free memory which has a prev (allocated "
                    "byl another memory and split previously).");
  DEBUG_LOG("%s: %zu at %p\n", __func__, this->bytes(), ptr_);
  cuda_set_device(device_num_);
  NBLA_CUDA_CHECK(cudaFree(ptr_));
}

bool CudaMemory::alloc_impl() {
  cuda_set_device(device_num_);
  try {
    NBLA_CUDA_CHECK(cudaMalloc(&ptr_, this->bytes()));
  } catch (...) {
    return false;
  }
  DEBUG_LOG("%s: %zu at %p (%d)\n", __func__, this->bytes(), ptr_, device_num_);
  return true;
}

shared_ptr<Memory> CudaMemory::divide_impl(size_t second_start) {
  constexpr int memory_alignment = 512;
  NBLA_FORCE_ASSERT(second_start % memory_alignment == 0,
                    "CUDA memory should be aligned with 512 bytes. Given %zu.",
                    second_start);
  size_t out_bytes = this->bytes() - second_start;
  void *out_ptr = (void *)((uint8_t *)ptr_ + second_start);
  return shared_ptr<Memory>(
      new CudaMemory(out_bytes, this->device_id(), out_ptr));
}

void CudaMemory::merge_next_impl(Memory *from) {}

void CudaMemory::merge_prev_impl(Memory *from) { ptr_ = from->pointer(); }
}
