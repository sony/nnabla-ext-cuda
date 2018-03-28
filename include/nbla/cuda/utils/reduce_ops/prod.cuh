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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_PROD_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_PROD_CUH__

#include <nbla/cuda/half.hpp>

#include <nbla/cuda/utils/types.cuh>

namespace nbla {

// ----------------------------------------------------------------------------
// Prod op
// ----------------------------------------------------------------------------
template <typename T> class ProdOp {
  const T *x_; // Reduction inputs.
  T *y_;       // Reduction outputs.
public:
  typedef typename CudaTypeForceFloat<T>::type storage_type;

  ProdOp(const T *x, T *y) : x_(x), y_(y) {}

  __forceinline__ __device__ void init(storage_type &thread_data) {
    thread_data = 1;
  }

  __forceinline__ __device__ storage_type premap(int i) { return x_[i]; }

  __forceinline__ __device__ void reduce(storage_type &to,
                                         const storage_type &from) {
    to *= from;
  }

  __forceinline__ __device__ void postmap(int j,
                                          const storage_type &thread_data) {
    y_[j] = thread_data;
  }
};
}
#endif