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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/memory/cuda_virtual_memory.hpp>

#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000

namespace nbla {

// ----------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------

void set_device_primary_ctx(int device_id) {
  static CUcontext prev_ctx;

  cuda_set_device(device_id);

  CUcontext ctx;
  NBLA_CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&ctx, device_id));

  if (ctx == prev_ctx)
    return;

  NBLA_CUDA_DRIVER_CHECK(cuCtxSetCurrent(ctx));
  prev_ctx = ctx;
}

CUmemAllocationProp &get_mem_allocation_prop(int device_id) {
  static unordered_map<int, CUmemAllocationProp> prop_map;

  if (prop_map.find(device_id) != prop_map.end())
    return prop_map[device_id];

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.win32HandleMetaData = NULL; // need to change on win32?

  return prop_map[device_id] = prop;
}

CUmemAccessDesc get_mem_access_desc(int device_id) {
  static unordered_map<int, CUmemAccessDesc> desc_map;

  if (desc_map.find(device_id) != desc_map.end())
    return desc_map[device_id];

  CUmemAccessDesc desc = {};
  desc.location = get_mem_allocation_prop(device_id).location;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  desc_map[device_id] = desc;

  return desc;
}

size_t get_allocation_granularity(int device_id) {
  static unordered_map<int, size_t> granularity_map;

  if (granularity_map.find(device_id) != granularity_map.end())
    return granularity_map[device_id];

  // Make sure to set ctx.
  set_device_primary_ctx(device_id);

  // get prop
  auto prop = get_mem_allocation_prop(device_id);

  // get granularity to round up to the valid chunk size.
  size_t chunk_size = 0;
  NBLA_CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(
      &chunk_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  granularity_map[device_id] = chunk_size;

  return chunk_size;
}

size_t round_up_by_chunk(size_t x, int device_id) {
  auto chunk_size = get_allocation_granularity(device_id);

  return (x + chunk_size - 1) / chunk_size * chunk_size;
}

// ----------------------------------------------------------------------
// CudaPhysicalMemory implementation
// ----------------------------------------------------------------------
CudaPhysicalMemory::~CudaPhysicalMemory() {
  if (allocated_)
    NBLA_CUDA_DRIVER_CHECK(cuMemRelease(handle_));
}

size_t CudaPhysicalMemory::alloc() {
  if (allocated_)
    return bytes_;

  int dev_id = std::stoi(device_id_);

  // make sure to set ctx.
  set_device_primary_ctx(dev_id);

  // Member bytes_ are updated by rounded bytes.
  bytes_ = round_up_by_chunk(bytes_, dev_id);

  try {
    static uint64_t count = 0;
    count++;

    // allocate physical memory
    auto &prop = get_mem_allocation_prop(dev_id);
    NBLA_CUDA_DRIVER_CHECK(cuMemCreate(&handle_, bytes_, &prop, 0ULL));

    allocated_ = true; // physical memory allocation is performed only once.

  } catch (...) {
    // release handle to avoid memory leak.
    NBLA_CUDA_DRIVER_CHECK(cuMemRelease(handle_));
    return 0;
  }

  return bytes_;
}

// ----------------------------------------------------------------------
// CudaVirtualMemory implementation
// ----------------------------------------------------------------------

CudaVirtualMemory::CudaVirtualMemory(size_t bytes, const string &device_id,
                                     VecPhysicalMemoryPtr p_memories)
    : Memory(bytes, device_id), event_{CudaEventFlag::DisableTiming} {
  NBLA_CHECK(bytes == round_up_by_chunk(bytes, std::stoi(device_id)),
             error_code::memory,
             "Bytes size passed is not a multiple of chunk size.");
  dev_ptr_ = 0ULL;
  memory_type_ = MemoryType::Virtual;
  p_memories_ = std::move(p_memories);
}

CudaVirtualMemory::~CudaVirtualMemory() {
  // free all virtual address
  free_virtual_address();
}

void CudaVirtualMemory::free_virtual_address() {
  if (dev_ptr_) {
    // Make sure to set ctx.
    set_device_primary_ctx(std::stoi(this->device_id()));

    // Unmap virtual address.
    NBLA_CUDA_DRIVER_CHECK(cuMemUnmap(dev_ptr_, this->bytes()));

    // Free virtual address.
    for (auto &e : va_ranges_) {
      NBLA_CUDA_DRIVER_CHECK(cuMemAddressFree(e.first, e.second));
    }
  }

  // reset members
  dev_ptr_ = 0ULL;
}

void CudaVirtualMemory::bind_impl() {
  // Calling bind_impl() more than once is prohibited, raise.
  // todo: support growing memory.
  if (dev_ptr_)
    NBLA_ERROR(error_code::memory,
               "Calling bind_impl() more than once is prohibited.");

  int d_id = std::stoi(this->device_id());

  // make sure to set ctx.
  set_device_primary_ctx(d_id);

  // Reserve virtual address.
  NBLA_CUDA_DRIVER_CHECK(
      cuMemAddressReserve(&dev_ptr_, this->bytes(), 0ULL, 0ULL, 0ULL));
  NBLA_CHECK(dev_ptr_ != 0ULL, error_code::memory, "allocation failed.");

  // Map virtual address to physical memory
  size_t mapped_bytes = 0;
  for (auto &m : p_memories_) {
    // Cast to CudaPhysicalMemory.
    auto pm = std::dynamic_pointer_cast<CudaPhysicalMemory>(m);

    // Make sure physical memory is already allocated.
    NBLA_CHECK(pm->alloc() == pm->bytes(), error_code::memory,
               "Physical memory allocation failed.");

    // Map virtual memory to a physical memory.
    NBLA_CUDA_DRIVER_CHECK(cuMemMap(dev_ptr_ + mapped_bytes, pm->bytes(), 0ULL,
                                    pm->get_handle(), 0ULL));

    mapped_bytes += pm->bytes();
  }

  auto accessDesc = get_mem_access_desc(d_id);
  NBLA_CUDA_DRIVER_CHECK(
      cuMemSetAccess(dev_ptr_, this->bytes(), &accessDesc, 1ULL));

  // Make ptr_ accessible.
  ptr_ = this->get_pointer();
  va_ranges_.emplace_back(dev_ptr_, this->bytes());
}

void CudaVirtualMemory::unbind_impl() {
  // todo: Freeing in destructor is enough?
  free_virtual_address();
}

bool CudaVirtualMemory::grow_impl(VecPhysicalMemoryPtr &p_mems) {
  if (p_mems.size() == 0)
    return true;

  // Calling bind_impl() more than once is prohibited, raise.
  // todo: support growing memory.
  int d_id = std::stoi(this->device_id());

  // make sure to set ctx.
  set_device_primary_ctx(d_id);

  // Reserve virtual address.
  CUdeviceptr new_ptr = 0ULL;
  size_t alloc_size = p_mems[0]->bytes() * p_mems.size();
  size_t reserved_size = p_memories_[0]->bytes() * p_memories_.size();
  CUresult status = cuMemAddressReserve(&new_ptr, alloc_size, 0ULL,
                                        dev_ptr_ + reserved_size, 0ULL);

  if (status != CUDA_SUCCESS || new_ptr != dev_ptr_ + reserved_size)
    return false;

  // Map virtual address to physical memory
  size_t mapped_bytes = reserved_size;
  for (auto &m : p_mems) {
    // Cast to CudaPhysicalMemory.
    auto pm = std::dynamic_pointer_cast<CudaPhysicalMemory>(m);

    // Make sure physical memory is already allocated.
    NBLA_CHECK(pm->alloc() == pm->bytes(), error_code::memory,
               "Physical memory allocation failed.");

    // Map virtual memory to a physical memory.
    NBLA_CUDA_DRIVER_CHECK(cuMemMap(dev_ptr_ + mapped_bytes, pm->bytes(), 0ULL,
                                    pm->get_handle(), 0ULL));

    mapped_bytes += pm->bytes();
  }

  NBLA_CHECK(mapped_bytes == reserved_size + alloc_size, error_code::memory,
             "memory size mismutch.");

  auto accessDesc = get_mem_access_desc(d_id);
  NBLA_CUDA_DRIVER_CHECK(
      cuMemSetAccess(dev_ptr_ + reserved_size, alloc_size, &accessDesc, 1ULL));

  // update members
  bytes_ += alloc_size;
  va_ranges_.emplace_back(new_ptr, alloc_size);

  return true;
}

DeviceMemoryState CudaVirtualMemory::get_device_memory_state() {
  cudaError_t status = event_.query();

  if (status == cudaSuccess)
    return DeviceMemoryState::Unlocked;
  else if (status == cudaErrorNotReady)
    return DeviceMemoryState::Locked;
  else
    NBLA_CUDA_CHECK(status); // raise by message
}

void CudaVirtualMemory::lock_device_memory() { event_.record(0); }
}

#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
