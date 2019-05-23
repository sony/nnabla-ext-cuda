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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/cuda.hpp>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <memory>
#include <vector>

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
  NBLA_CUDA_CHECK(cudaMemset(this->pointer<void>(), 0,
                             this->size() * sizeof_dtype(this->dtype_)));
}

Context CudaArray::filter_context(const Context &ctx) {
  return Context({}, "CudaArray", ctx.device_id);
}

/////////////////////////////////////
// Register cpu --> cuda synchronizer
/////////////////////////////////////
void synchronizer_cuda_array_cpu_array(Array *src, Array *dst) {
  if (src->dtype() != dst->dtype()) {
    // if dtype mismatches, transfer gpu-cpu first, then convert dtype.
    Context ctx = dst->context();
    unique_ptr<Array> tmp(new CpuCachedArray(src->size(), src->dtype(), ctx));
    synchronizer_cuda_array_cpu_array(src, tmp.get());
    dst->copy_from(tmp.get());
    return;
  }
  size_t size = src->size() * sizeof_dtype(dst->dtype());
  // Ensure it runs on devices which doesn't support unified virtual addressing.
  cuda_set_device(std::stoi(src->context().device_id));
  NBLA_CUDA_CHECK(cudaMemcpy(dst->pointer<void>(), src->const_pointer<void>(),
                             size, cudaMemcpyDeviceToHost));
}

/////////////////////////////////////
// Register cpu --> cuda synchronizer
/////////////////////////////////////
void synchronizer_cpu_array_cuda_array(Array *src, Array *dst) {
  if (src->dtype() != dst->dtype()) {
    // If dtype mismatches, transfer cpu-gpu first, then convert dtype in gpu.
    Context ctx = dst->context();
    unique_ptr<Array> tmp(new CudaCachedArray(src->size(), src->dtype(), ctx));
    synchronizer_cpu_array_cuda_array(src, tmp.get());
    dst->copy_from(tmp.get());
    return;
  }
  size_t size = src->size() * sizeof_dtype(dst->dtype());
  // Ensure it runs on devices which doesn't support unified virtual addressing.
  cuda_set_device(std::stoi(dst->context().device_id));
  NBLA_CUDA_CHECK(cudaMemcpy(dst->pointer<void>(), src->const_pointer<void>(),
                             size, cudaMemcpyHostToDevice));
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

} // End of namespace nbla
