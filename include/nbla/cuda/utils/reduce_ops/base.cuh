// Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_CUDA_UTILS_REDUCE_OPS_BASE_CUH__
#define __NBLA_CUDA_UTILS_REDUCE_OPS_BASE_CUH__

namespace nbla {

/** Interface class of reduction operators.
 */
template <class ReduceOpTypes> class ReduceOpBase {
public:
  // Types used in reduction CUDA kernel provided by ReduceOpTypes.
  // See also nnabla-ext-cuda/include/nbla/cuda/utils/types.cuh.
  using Types = ReduceOpTypes;

  // Just shorten the type names for code readability.
  // These name declarations are required as the interface.
  using Tcu = typename Types::Tcu;
  using IndexT = typename Types::IndexT;
  using StorageT = typename Types::StorageT;

  // Input buffer
  const Tcu *const input;

  // Temporary buffer
  StorageT *buf;

protected:
  // Output buffers
  Tcu *const output_; // output values
  Size_t *const idx_; // input indeces used for e.g., MaxCuda.

public:
  ReduceOpBase(const Tcu *const in, Tcu *const out, Size_t *const idx)
      : input(in), output_(out), idx_(idx) {}
  virtual __device__ StorageT init() = 0;
  virtual __device__ StorageT make_storage(const Tcu v, const IndexT idx) = 0;
  virtual __device__ StorageT operator()(const StorageT &a,
                                         const StorageT &b) = 0;
  virtual __device__ void store(const IndexT idx, const StorageT &v) = 0;
  virtual __device__ void intermediate_store(const IndexT idx,
                                             const StorageT &v) = 0;
};
}
#endif