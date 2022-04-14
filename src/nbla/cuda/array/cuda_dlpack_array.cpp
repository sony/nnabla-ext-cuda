// Copyright 2020,2021 Sony Corporation.
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

#include <nbla/cuda/array/cuda_dlpack_array.hpp>
#include <nbla/cuda/common.hpp>

namespace nbla {

CudaDlpackArray::CudaDlpackArray(const Size_t size, dtypes dtype,
                                 const Context &ctx,
                                 const AllocatorMemoryPtr mem,
                                 const Size_t offset)
    : DlpackArray(size, dtype, ctx, mem, offset) {}

CudaDlpackArray::~CudaDlpackArray() {}

Context CudaDlpackArray::filter_context(const Context &ctx) {
  return Context({}, "CudaDlpackArray", ctx.device_id);
}

void CudaDlpackArray::zero() {
  cuda_set_device(device_);
  NBLA_CUDA_CHECK(cudaMemset(this->pointer<void>(), 0,
                             this->size() * sizeof_dtype(this->dtype_)));
}
} // End of namespace nbla
