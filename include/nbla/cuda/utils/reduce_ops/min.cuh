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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_MIN_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_MIN_CUH__

#include <nbla/cuda/utils/reduce_ops/max.cuh>

namespace nbla {

// ----------------------------------------------------------------------------
// Base class
// ----------------------------------------------------------------------------
template <typename T> class BaseMinOp : public BaseMaxOp<T> {
public:
  typedef floatint storage_type;
  BaseMinOp(T *val, int *ind) : BaseMaxOp<T>(val, ind) {}

  __forceinline__ __device__ void init(storage_type &thread_data) {
    thread_data.f = +1e+8;
    thread_data.i = 0;
  }

  __forceinline__ __device__ void reduce(storage_type &to,
                                         const storage_type &from) {
    if (from.f < to.f) {
      to.f = from.f;
      to.i = from.i;
    }
  }
};

// ----------------------------------------------------------------------------
// Per-block min
// ----------------------------------------------------------------------------
template <typename T> class MinPreOp : public BaseMinOp<T> {
protected:
  const T *x_;

public:
  typedef floatint storage_type;
  MinPreOp(const T *x, T *val, int *ind) : x_(x), BaseMinOp<T>(val, ind) {}

  __forceinline__ __device__ storage_type premap(int i) {
    return storage_type{x_[i], i};
  }
};

// ----------------------------------------------------------------------------
// Finalize
// ----------------------------------------------------------------------------
template <typename T> class MinPostOp : public MinPreOp<T> {
protected:
  // Block min and index
  const int *bind_;

public:
  typedef floatint storage_type;
  MinPostOp(const T *x, const int *bind, T *val, int *ind)
      : MinPreOp<T>(x, val, ind), bind_(bind) {}

  __forceinline__ __device__ storage_type premap(int i) {
    return storage_type{this->x_[i], bind_[i]};
  }
};
}
#endif