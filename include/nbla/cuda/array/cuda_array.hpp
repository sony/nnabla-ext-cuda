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

#ifndef __NBLA_CUDA_ARRAY_HPP__
#define __NBLA_CUDA_ARRAY_HPP__

#include <memory>

#include <cuda.h>
#include <nbla/array.hpp>
#include <nbla/array/cpu_array.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/synced_array.hpp>

namespace nbla {

using std::shared_ptr;

/** Array on CUDA devices.
\ingroup ArrayImplGrp
*/
class CudaArray : public Array {
protected:
  int device_;

public:
  explicit CudaArray(const Size_t size, dtypes dtype, const Context &ctx);
  explicit CudaArray(const Size_t size, dtypes dtype, const Context &ctx,
                     AllocatorMemory &&mem);
  virtual ~CudaArray();
  virtual void copy_from(const Array *src_array);
  virtual void zero();
  virtual void fill(float value);
  static Context filter_context(const Context &ctx);
};

NBLA_CUDA_API void
synchronizer_cuda_array_cpu_array(Array *src, Array *dst,
                                  const int async_flags = AsyncFlag::NONE);

NBLA_CUDA_API void
synchronizer_cpu_array_cuda_array(Array *src, Array *dst,
                                  const int async_flags = AsyncFlag::NONE);

/** Array allocated on CUDA device with a CudaMemory obtained by
Cuda::caching_allocator().
*/
class CudaCachedArray : public CudaArray {
public:
  /** Constructor

      @param size Length of array.
      @param dtype Data type.
      @param ctx Context specifies device ID.
   */
  explicit CudaCachedArray(const Size_t size, dtypes dtype, const Context &ctx);
  virtual ~CudaCachedArray();
  static Context filter_context(const Context &ctx);
};

/** Array allocated on unified memory with a CudaUnifiedMemory obtained by
Cuda::unified_allocator().
*/
class CudaCachedUnifiedArray : public CudaArray {
public:
  /** Constructor

  @param size Length of array.
  @param dtype Data type.
  @param ctx Context specifies device ID.
  */
  explicit CudaCachedUnifiedArray(const Size_t size, dtypes dtype,
                                  const Context &ctx);
  virtual ~CudaCachedUnifiedArray();
  static Context filter_context(const Context &ctx);
};

/** Array allocated on host with a CudaHostMemory obtained by
Cuda::pinned_allocator().
*/
class CudaCachedHostArray : public CpuArray {
public:
  /** Constructor

  @param size Length of array.
  @param dtype Data type.
  @param ctx Context.
  */
  explicit CudaCachedHostArray(const Size_t size, dtypes dtype,
                               const Context &ctx);
  virtual ~CudaCachedHostArray();
  static Context filter_context(const Context &ctx);
};

#if CUDA_VERSION >= 10020
/** Array allocated on CUDA device with virtual memory management obtained by
 * Cuda::virtual_caching_allocator().
 */
class CudaCachedVirtualArray : public CudaArray {
public:
  /** Constructor

  @param size Length of array.
  @param dtype Data type.
  @param ctx Context.
  */

  shared_ptr<Allocator> select_allocator(const size_t size,
                                         const string &device_id);

  explicit CudaCachedVirtualArray(const Size_t size, dtypes dtype,
                                  const Context &ctx);
  virtual ~CudaCachedVirtualArray();
  static Context filter_context(const Context &ctx);
};
#endif // CUDA_VERSION >= 10020
}
#endif
