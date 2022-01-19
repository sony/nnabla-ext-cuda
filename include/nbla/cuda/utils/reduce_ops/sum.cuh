// Copyright 2021 Sony Group Corporation.
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

#include <nbla/cuda/utils/fast_reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/base.cuh>

namespace nbla {

/** Reduction operator to compute sum.

    Template parameters
      - T: the type of the input and output values.
      - U: the type of the size, shape, and indices of the input and output.
 */
template <class T, class U>
class ReduceOpSum : public ReduceOpBase<ReduceOpSumLikeType<T, U>> {
public:
  using Types = ReduceOpSumLikeType<T, U>;
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  ReduceOpSum(const Tcu *const in, Tcu *const out)
      : ReduceOpBase<ReduceOpSumLikeType<T, U>>(in, out, nullptr) {}

  __device__ StorageT make_storage(const Tcu v, const IndexT idx) override {
    return StorageT(v);
  }

  __device__ StorageT init() override { return StorageT(0); }

  __device__ StorageT operator()(const StorageT &a,
                                 const StorageT &b) override {
    return a + b;
  }

  __device__ void store(const IndexT idx, const StorageT &v) override {
    this->output_[idx] = v;
  }

  __device__ void intermediate_store(const IndexT idx,
                                     const StorageT &v) override {
    this->buf[idx] = v;
  }
};

/** The sum of x is computed on GPU according to the setup parameters in
    reduce_setup. The results are stored into y.
 */
template <class T>
void device_sum(const Context &ctx, const T *const x, T *const y,
                const ReduceSetup &reduce_setup) {
  if (reduce_setup.require_64bit_index) {
    fast_reduce(ctx, ReduceOpSum<T, Size_t>(x, y), reduce_setup);
  } else {
    fast_reduce(ctx, ReduceOpSum<T, uint32_t>(x, y), reduce_setup);
  }
}
}
#endif