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

#include <nbla/array.hpp>
#include <nbla/cuda/cuda_memory.hpp>
#include <nbla/cuda/defs.hpp>

namespace nbla {

using std::shared_ptr;

/** Array on CUDA devices.
\ingroup ArrayImplGrp
*/
class CudaArray : public Array {
protected:
  int device_;
  /* Holding CudaMemory until the instance is destroyed to prevent freeing.
   */
  shared_ptr<CudaMemory> inuse_memory_;

public:
  explicit CudaArray(const Size_t size, dtypes dtype, const Context &ctx);
  virtual ~CudaArray();
  virtual void copy_from(const Array *src_array);
  virtual void zero();
  virtual void fill(float value);
  static Context filter_context(const Context &ctx);

protected:
  virtual void allocate();
  virtual void deallocate();
};

NBLA_CUDA_API void synchronizer_cuda_array_cpu_array(Array *src, Array *dst);

NBLA_CUDA_API void synchronizer_cpu_array_cuda_array(Array *src, Array *dst);

/** Array allocated on CUDA device with Memory Pool

This is a necessary ingredient for imperative programming interface of
neural networks (aka define-by-run or dynamic). Memory allocation of
CUDA is not asynchronous. Hence, allocating memory region between each
function will lead thread synchronization that will block executions of
CUDA kernels. Then, your network execution will be slow and inefficient.
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

protected:
  virtual void allocate();
  virtual void deallocate();
};
}
#endif
