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

#ifndef __NBLA_CUDA_UTILS_POINTERS_CUH__
#define __NBLA_CUDA_UTILS_POINTERS_CUH__

#include <nbla/cuda/array/cuda_array.hpp>

namespace nbla {

template <typename T>
shared_ptr<CudaCachedArray>
get_cuda_pointer_array(const vector<Variable *> &inputs, const Context &ctx,
                       std::function<const T *(int)> getter) {
  size_t bytes = inputs.size() * sizeof(T *);

  // Create CPU array and set pointers from variables.
  std::unique_ptr<const T *> xptrs_cpu(new const T *[inputs.size()]);
  const T **xptrs_cpu_raw = xptrs_cpu.get();
  for (int i = 0; i < inputs.size(); ++i) {
    xptrs_cpu_raw[i] = getter(i);
  }

  // Create CUDA array and copy the addresses from CPU to GPU.
  auto xptrs_array = make_shared<CudaCachedArray>(bytes, dtypes::BYTE, ctx);
  const T **xptrs = xptrs_array->template pointer<const T *>();
  NBLA_CUDA_CHECK(
      cudaMemcpy(xptrs, xptrs_cpu_raw, bytes, cudaMemcpyHostToDevice));

  // Return CudaCachedArray.
  return xptrs_array;
}

template <typename SRC_T, typename DST_T = SRC_T>
shared_ptr<NdArray> create_ndarray_from_vector(const vector<SRC_T> &src) {
  Context cpu_ctx({"cpu::float"}, "CpuCachedArray");
  int size = src.size();
  auto ret = make_shared<NdArray>(Shape_t{size});
  DST_T *cpu_ptr =
      ret->cast(get_dtype<DST_T>(), cpu_ctx, true)->pointer<DST_T>();
  std::copy_n(src.cbegin(), src.size(), cpu_ptr);
  return ret;
}
}
#endif
