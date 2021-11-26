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

#ifndef __NBLA_CUDA_UTILS_SCAN_OPS_BASE_CUH__
#define __NBLA_CUDA_UTILS_SCAN_OPS_BASE_CUH__

namespace nbla {

/** Interface class of scan operators.
 */
template <class ScanOpTypes> class ScanOpBase {
public:
  // Types used in reduction CUDA kernel provided by ScanOpTypes.
  // See also nnabla-ext-cuda/include/nbla/cuda/utils/types.cuh.
  using Types = ScanOpTypes;

  // Just shorten the type names for code readability.
  // These name declarations are required as the interface.
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  // Input buffer
  const Tcu *const input;

  // Temporary buffer
  StorageT *buf;

  // Output buffers
  Tcu *const output;

public:
  ScanOpBase(const Tcu *const in, Tcu *const out) : input(in), output(out) {}
  virtual __device__ StorageT init() = 0;
  virtual __device__ StorageT make_storage(const Tcu v) = 0;
  virtual __device__ StorageT operator()(const StorageT &acc,
                                         const StorageT &v) = 0;
  template <bool accum>
  __device__ void store(const IndexT idx, const StorageT &v) {
    output[idx] = v + (accum ? output[idx] : (Tcu)0);
  }
  virtual __device__ void intermediate_store(const IndexT idx,
                                             const StorageT &v) = 0;
};
}

#endif