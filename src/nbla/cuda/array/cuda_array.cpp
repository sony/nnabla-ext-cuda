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

#include <nbla/array/cpu_array.hpp>
#include <nbla/array_registry.hpp>
#include <nbla/cpu.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/event.hpp>
#include <nbla/cuda/function/my_cuda_memset.hpp>
#include <nbla/nd_array.hpp>
#include <nbla/singleton_manager.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <memory>
#include <vector>

#include <nbla/cuda/memory/cuda_virtual_memory.hpp>
#include <nbla/memory/virtual_caching_allocator.hpp>

namespace nbla {

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;

// CudaArray
CudaArray::CudaArray(const Size_t size, dtypes dtype, const Context &ctx)
    : Array(size, dtype, ctx,
            SingletonManager::get<Cuda>()->naive_allocator()->alloc(
                Array::size_as_bytes(size, dtype), ctx.device_id)),
      device_(std::stoi(ctx.device_id)) {}

CudaArray::CudaArray(const Size_t size, dtypes dtype, const Context &ctx,
                     AllocatorMemory &&mem)
    : Array::Array(size, dtype, ctx, std::move(mem)),
      device_(std::stoi(ctx.device_id)) {}

CudaArray::~CudaArray() {}

void CudaArray::zero() {
  cuda_set_device(device_);
  /* cudaMemset and cudaMemsetAsync issued into null stream
     block a non-blocking stream. It seems to be a bug of CUDA.

     Version info:

     nvcc: NVIDIA (R) Cuda compiler driver
     Copyright (c) 2005-2019 NVIDIA Corporation
     Built on Fri_Feb__8_19:08:26_Pacific_Standard_Time_2019
     Cuda compilation tools, release 10.1, V10.1.105
  */
  // NBLA_CUDA_CHECK(cudaMemset(this->pointer<void>(), 0,
  //                           this->size() * sizeof_dtype(this->dtype_)));

  my_cudaMemset(this->pointer<void>(), 0,
                this->size() * sizeof_dtype(this->dtype_));
}

Context CudaArray::filter_context(const Context &ctx) {
  return Context({}, "CudaArray", ctx.device_id);
}

void CUDART_CB delete_callback(cudaStream_t stream, cudaError_t status,
                               void *userData) {
  delete reinterpret_cast<shared_ptr<Array> *>(userData);
}

// Main process of asynchronous synchronizer
void synchronize_async_cuda_array_cpu_array(Array *src, Array *dst,
                                            cudaMemcpyKind kind,
                                            const cudaStream_t stream,
                                            const int async_flags) {
  // Wait an previous asynchronous memcpy
  src->wait_event(dst->context(), async_flags);

  if (dst->have_event()) {
    NBLA_ERROR(error_code::target_specific_async,
               "Duplicated asynchronous memcpy to the same destination array");
  }

  cudaEvent_t null_event;
  NBLA_CUDA_CHECK(cudaEventCreate(&null_event));
  NBLA_CUDA_CHECK(cudaEventRecord(null_event, 0));
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(stream, null_event, 0));
  NBLA_CUDA_CHECK(cudaEventDestroy(null_event));

  // Prepare an event
  cudaEvent_t event;
  NBLA_CUDA_CHECK(cudaEventCreate(&event));

  // Memory copy
  size_t size = src->size() * sizeof_dtype(dst->dtype());
  NBLA_CUDA_CHECK(cudaMemcpyAsync(
      dst->pointer<void>(), src->const_pointer<void>(), size, kind, stream));

  // No cudaStreamCallback becasuse cudaStreamSycnhronize(0) in
  // CudaEvent::wait_event
  // has the same effect.

  // Record the memory copy as an event into the destination array
  NBLA_CUDA_CHECK(cudaEventRecord(event, stream));
  dst->set_event(shared_ptr<Event>(new CudaEvent(event, src->getptr())));
}

// Main process of asynchronous synchronizer
void synchronize_async_cpu_array_cuda_array(Array *src, Array *dst,
                                            cudaMemcpyKind kind,
                                            const cudaStream_t stream,
                                            const int async_flags) {
  // Wait an previous asynchronous memcpy
  src->wait_event(dst->context(), async_flags);

  if (dst->have_event()) {
    NBLA_ERROR(error_code::target_specific_async,
               "Duplicated asynchronous memcpy to the same destination array");
  }

  // Synchronize to null stream which managed the GPU memory usage of the array
  // dst
  cudaEvent_t null_event;
  NBLA_CUDA_CHECK(cudaEventCreate(&null_event));
  NBLA_CUDA_CHECK(cudaEventRecord(null_event, 0));
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(stream, null_event, 0));
  NBLA_CUDA_CHECK(cudaEventDestroy(null_event));

  // Prepare an event
  cudaEvent_t event;
  NBLA_CUDA_CHECK(cudaEventCreate(&event));

  // Memory copy
  size_t size = src->size() * sizeof_dtype(dst->dtype());
  NBLA_CUDA_CHECK(cudaMemcpyAsync(
      dst->pointer<void>(), src->const_pointer<void>(), size, kind, stream));

  if (!(async_flags & AsyncFlag::UNSAFE)) { // Keep safe CPU memory of src
    auto delete_guard = new shared_ptr<Array>(src->getptr());
    NBLA_CUDA_CHECK(
        cudaStreamAddCallback(stream, delete_callback, delete_guard, 0));
  }

  // Record the memory copy as an event into the destination array
  NBLA_CUDA_CHECK(cudaEventRecord(event, stream));
  dst->set_event(shared_ptr<Event>(new CudaEvent(event, src->getptr())));
}

// Main process of synchronous synchronizer
void synchronize_sync(Array *src, Array *dst, cudaMemcpyKind kind,
                      const int async_flags) {
  // Wait an previous asynchronous memcpy
  src->wait_event(dst->context(), async_flags);

  if (dst->have_event()) {
    NBLA_ERROR(error_code::target_specific_async,
               "Duplicated asynchronous memcpy to the same destination array");
  }

  // Memory copy
  size_t size = src->size() * sizeof_dtype(dst->dtype());
  NBLA_CUDA_CHECK(
      cudaMemcpy(dst->pointer<void>(), src->const_pointer<void>(), size, kind));
  // Record no event because memory copy has been finished synchronously
  dst->set_event(nullptr);
}

/////////////////////////////////////
// Register cuda --> cpu synchronizer
/////////////////////////////////////
void synchronizer_cuda_array_cpu_array(Array *src, Array *dst,
                                       const int async_flags) {
  // Ensure it runs on devices which doesn't support unified virtual addressing.
  cuda_set_device(std::stoi(src->context().device_id));

  if (src->dtype() != dst->dtype()) {
    // if dtype mismatches, convert dtype first, and then transfer gpu-cpu.
    NdArray tmp_arr(Shape_t{src->size()});
    // cast() in synchronizer passes the off-recording flag for LMS;
    Array *tmp = tmp_arr.array()->cast(dst->dtype(), src->context(), true,
                                       AsyncFlag::OFFREC);
    src->wait_event(tmp->context(), async_flags);
    tmp->copy_from(src);
    synchronizer_cuda_array_cpu_array(tmp, dst, async_flags);
    return;
  }

  if (async_flags & AsyncFlag::ASYNC) { // cudaMemcpyAsync
    synchronize_async_cuda_array_cpu_array(
        src, dst, cudaMemcpyDeviceToHost,
        SingletonManager::get<Cuda>()->stream_DtoH, async_flags);
  } else { // cudaMemcpy
    synchronize_sync(src, dst, cudaMemcpyDeviceToHost, async_flags);
  }
}

/////////////////////////////////////
// Register cpu --> cuda synchronizer
/////////////////////////////////////
void synchronizer_cpu_array_cuda_array(Array *src, Array *dst,
                                       const int async_flags) {
  // Ensure it runs on devices which doesn't support unified virtual addressing.
  cuda_set_device(std::stoi(dst->context().device_id));

  if (src->dtype() != dst->dtype()) {
    // If dtype mismatches, transfer cpu-gpu first, then convert dtype in gpu.
    NdArray tmp_arr(Shape_t{src->size()});
    // cast() in synchronizer passes the off-recording flag for LMS;
    Array *tmp = tmp_arr.array()->cast(src->dtype(), dst->context(), true,
                                       AsyncFlag::OFFREC);
    synchronizer_cpu_array_cuda_array(src, tmp, async_flags);
    tmp->wait_event(dst->context(), async_flags);
    dst->copy_from(tmp);
    return;
  }

  if (async_flags & AsyncFlag::ASYNC) { // cudaMemcpyAsync
    synchronize_async_cpu_array_cuda_array(
        src, dst, cudaMemcpyHostToDevice,
        SingletonManager::get<Cuda>()->stream_HtoD, async_flags);
  } else { // cudaMemcpy
    synchronize_sync(src, dst, cudaMemcpyHostToDevice, async_flags);
  }
}

/////////////////////////////////
// CudaCachedArray implementation
/////////////////////////////////
CudaCachedArray::CudaCachedArray(const Size_t size, dtypes dtype,
                                 const Context &ctx)
    : CudaArray(size, dtype, ctx,
                SingletonManager::get<Cuda>()->caching_allocator()->alloc(
                    Array::size_as_bytes(size, dtype), ctx.device_id)) {}

CudaCachedArray::~CudaCachedArray() {}

Context CudaCachedArray::filter_context(const Context &ctx) {
  return Context({}, "CudaCachedArray", ctx.device_id);
}

////////////////////////////////////////
// CudaCachedUnifiedArray implementation
////////////////////////////////////////
CudaCachedUnifiedArray::CudaCachedUnifiedArray(const Size_t size, dtypes dtype,
                                               const Context &ctx)
    : CudaArray(size, dtype, ctx,
                SingletonManager::get<Cuda>()->unified_allocator()->alloc(
                    Array::size_as_bytes(size, dtype), ctx.device_id)) {}

CudaCachedUnifiedArray::~CudaCachedUnifiedArray() {}

Context CudaCachedUnifiedArray::filter_context(const Context &ctx) {
  return Context({}, "CudaCachedUnifiedArray", ctx.device_id);
}

/////////////////////////////////////
// CudaCachedHostArray implementation
/////////////////////////////////////
CudaCachedHostArray::CudaCachedHostArray(const Size_t size, dtypes dtype,
                                         const Context &ctx)
    : CpuArray(size, dtype, ctx,
               SingletonManager::get<Cuda>()->pinned_allocator()->alloc(
                   Array::size_as_bytes(size, dtype), "")) {}

CudaCachedHostArray::~CudaCachedHostArray() {}

Context CudaCachedHostArray::filter_context(const Context &ctx) {
  return Context({}, "CudaCachedHostArray", "");
}

#if CUDA_VERSION >= 10020
/////////////////////////////////////
// CudaCachedVirtualArray implementation
/////////////////////////////////////

shared_ptr<Allocator>
CudaCachedVirtualArray::select_allocator(const size_t size,
                                         const string &device_id) {
  auto cuda = SingletonManager::get<Cuda>();

  // size_t min_chunk_size = get_allocation_granularity(stoi(device_id));
  size_t min_chunk_size =
      std::dynamic_pointer_cast<VirtualCachingAllocatorBase>(
          cuda->virtual_caching_allocator())
          ->chunk_size_;

  return size <= min_chunk_size ? cuda->caching_allocator()
                                : cuda->virtual_caching_allocator();
}

CudaCachedVirtualArray::CudaCachedVirtualArray(const Size_t size, dtypes dtype,
                                               const Context &ctx)
    : CudaArray(
          size, dtype, ctx,
          // select_allocator(Array::size_as_bytes(size, dtype), ctx.device_id)->alloc(
        SingletonManager::get<Cuda>()->virtual_caching_allocator()->alloc(
                  Array::size_as_bytes(size, dtype), ctx.device_id)) {}

CudaCachedVirtualArray::~CudaCachedVirtualArray() {}

Context CudaCachedVirtualArray::filter_context(const Context &ctx) {
  return Context({}, "CudaCachedVirtualArray", ctx.device_id);
}
#endif // CUDA_VERSION >= 10020

} // End of namespace nbla
