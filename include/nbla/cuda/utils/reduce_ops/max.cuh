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

#include <nbla/cuda/limits.hpp>
#include <nbla/cuda/utils/fast_reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/base.cuh>
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

/** Reduction operator to compute max.

    Template parameters
      - T: the type of the input and output values.
      - U: the type of the size, shape, and indices of the input and output.
 */
template <class T, class U>
class ReduceOpMax : public ReduceOpBase<ReduceOpMaxLikeType<T, U>> {
public:
  using Types = ReduceOpMaxLikeType<T, U>;
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  ReduceOpMax(const Tcu *const in, Tcu *const out, Size_t *const idx)
      : ReduceOpBase<ReduceOpMaxLikeType<T, U>>(in, out, idx) {}

  __device__ StorageT init() override {
    return StorageT(-numeric_limits_cuda<Tcu>::max(), 0);
  }

  __device__ StorageT make_storage(const Tcu v, const IndexT idx) override {
    return StorageT(v, idx);
  }

  __device__ StorageT operator()(const StorageT &a,
                                 const StorageT &b) override {
    if (a.val > b.val) {
      return a;
    } else if (a.val == b.val) {
      if (a.idx < b.idx) {
        return a;
      } else {
        return b;
      }
    } else {
      return b;
    }
  }

  __device__ void store(const IndexT idx, const StorageT &v) override {
    this->output_[idx] = v.val;
    this->idx_[idx] = v.idx;
  }

  __device__ void intermediate_store(const IndexT idx,
                                     const StorageT &v) override {
    this->buf[idx] = v;
  }
};

/** The sum of x is computed on GPU according to the setup parameters in
    reduce_setup. The results are stored into y and idx.
 */
template <class T>
void device_max(const Context &ctx, const T *const x, T *const y,
                Size_t *const idx, const ReduceSetup &reduce_setup) {
  if (reduce_setup.require_64bit_index) {
    fast_reduce(ctx, ReduceOpMax<T, Size_t>(x, y, idx), reduce_setup);
  } else {
    fast_reduce(ctx, ReduceOpMax<T, uint32_t>(x, y, idx), reduce_setup);
  }
}
}
#endif