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

#ifndef __NBLA_CUDA_UTILS_SCAN_OPS_PROD_CUH__
#define __NBLA_CUDA_UTILS_SCAN_OPS_PROD_CUH__

#include <nbla/cuda/utils/scan.cuh>
#include <nbla/cuda/utils/scan_ops/base.cuh>
#include <nbla/cuda/utils/types.cuh>

namespace nbla {

/**
 * @brief Scan operator to compute cumulative prod.
 *
 * @tparam T the type of the input and output values;
 * @tparam U the type of the size, shape, and indices of the input and output.
 */
template <class T, class U>
class ScanOpProd : public ScanOpBase<ScanOpProdType<T, U>> {
public:
  using Types = ScanOpProdType<T, U>;
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  ScanOpProd(const Tcu *const in, Tcu *const out)
      : ScanOpBase<Types>(in, out) {}

  __device__ StorageT init() override { return StorageT(1); }

  __device__ StorageT make_storage(const Tcu v) override { return StorageT(v); }

  __device__ StorageT operator()(const StorageT &accum,
                                 const StorageT &v) override {
    return accum * v;
  }

  __device__ void intermediate_store(const IndexT idx,
                                     const StorageT &v) override {
    this->buf[idx] = v;
  }
};

/**
 * @brief The cumulative prod of x is computed on GPU according to the setup
 * parameters in scan_setup. The results are stored into y.
 */
template <class T>
void device_cumprod(const Context &ctx, const T *const x, T *const y,
                    const ScanSetup &scan_setup, const bool accum) {
  if (scan_setup.require_64bit_index) {
    ScanOpProd<T, Size_t> op(x, y);
    scan(ctx, op, scan_setup, accum);
  } else {
    ScanOpProd<T, int32_t> op(x, y);
    scan(ctx, op, scan_setup, accum);
  }
}
} // namespace nbla

#endif