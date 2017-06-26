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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_MAX_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_MAX_CUH__

#include <nbla/cuda/utils/types.cuh>

namespace nbla {

// ----------------------------------------------------------------------------
// Base class
// ----------------------------------------------------------------------------
template <typename T> class BaseMaxOp {
  T *val_;   // Reduction outputs.
  int *ind_; // Max indeces.
public:
  typedef floatint storage_type;

  BaseMaxOp(T *val, int *ind) : val_(val), ind_(ind) {}

  __forceinline__ __device__ void init(storage_type &thread_data) {
    thread_data.f = -1e+8;
    thread_data.i = 0;
  }

  __forceinline__ __device__ void reduce(storage_type &to,
                                         const storage_type &from) {
    if (from.f > to.f) {
      to.f = from.f;
      to.i = from.i;
    }
  }

  __forceinline__ __device__ void postmap(int j,
                                          const storage_type &thread_data) {
    val_[j] = thread_data.f;
    ind_[j] = thread_data.i;
  }
};

// ----------------------------------------------------------------------------
// Per-block max
// ----------------------------------------------------------------------------
template <typename T> class MaxPreOp : public BaseMaxOp<T> {
protected:
  const T *x_;

public:
  typedef floatint storage_type;
  MaxPreOp(const T *x, T *val, int *ind) : x_(x), BaseMaxOp<T>(val, ind) {}

  __forceinline__ __device__ storage_type premap(int i) {
    return storage_type{x_[i], i};
  }
};

// ----------------------------------------------------------------------------
// Finalize
// ----------------------------------------------------------------------------
template <typename T> class MaxPostOp : public MaxPreOp<T> {
protected:
  // Block max and index
  const int *bind_;

public:
  typedef floatint storage_type;
  MaxPostOp(const T *x, const int *bind, T *val, int *ind)
      : MaxPreOp<T>(x, val, ind), bind_(bind) {}

  __forceinline__ __device__ storage_type premap(int i) {
    return storage_type{this->x_[i], bind_[i]};
  }
};
}
#endif