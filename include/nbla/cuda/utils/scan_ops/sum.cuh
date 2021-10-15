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

#ifndef __NBLA_CUDA_UTILS_SCAN_OPS_SUM_CUH__
#define __NBLA_CUDA_UTILS_SCAN_OPS_SUM_CUH__

#include <nbla/cuda/utils/scan.cuh>
#include <nbla/cuda/utils/scan_ops/base.cuh>
#include <nbla/cuda/utils/types.cuh>

namespace nbla {

template <class T, class U>
class ScanOpSum : public ScanOpBase<ScanOpSumType<T, U>> {
public:
  using Types = ScanOpSumType<T, U>;
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  ScanOpSum(const Tcu *const in, Tcu *const out) : ScanOpBase<Types>(in, out) {}

  __device__ StorageT init() override { return StorageT(0); }

  __device__ StorageT make_storage(const Tcu v) override { return StorageT(v); }

  __device__ StorageT operator()(const StorageT &acc,
                                 const StorageT &v) override {
    return acc + v;
  }

  __device__ void intermediate_store(const IndexT idx,
                                     const StorageT &v) override {
    this->buf[idx] = v;
  }
};

template <class T>
void device_cumsum(const Context &ctx, const T *const x, T *const y,
                   const ScanSetup &scan_setup) {
  if (scan_setup.require_64bit_index) {
    scan(ctx, ScanOpSum<T, Size_t>(x, y), scan_setup);
  } else {
    scan(ctx, ScanOpSum<T, uint32_t>(x, y), scan_setup);
  }
}
}

#endif