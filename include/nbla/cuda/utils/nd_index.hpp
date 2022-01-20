// Copyright 2020,2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#ifndef __NBLA_CUDA_UTILS_ND_INDEX_HPP__
#define __NBLA_CUDA_UTILS_ND_INDEX_HPP__

#include <nbla/common.hpp>

namespace nbla {

template <int NDIM> struct NdIndex { int64_t nd_idx[NDIM]; };

template <int NDIM>
__device__ NdIndex<NDIM> device_flat2nd(int64_t idx, const int64_t *stride) {
  NdIndex<NDIM> nd_index;
  for (int i = 0; i < NDIM; i++) {
    nd_index.nd_idx[i] = idx / stride[i];
    idx -= nd_index.nd_idx[i] * stride[i];
  }
  return nd_index;
}

template <int NDIM>
__device__ int64_t device_nd2flat(NdIndex<NDIM> &nd_index,
                                  const int64_t *stride) {
  auto idx = 0;
  for (int i = 0; i < NDIM; i++) {
    idx += stride[i] * nd_index.nd_idx[i];
  }
  return idx;
}

template <int NDIM>
__device__ NdIndex<NDIM> device_flat_to_nd(int64_t idx,
                                           const NdIndex<NDIM> strides) {
  NdIndex<NDIM> nd_index;
  for (int i = 0; i < NDIM; i++) {
    nd_index.nd_idx[i] = idx / strides.nd_idx[i];
    idx -= nd_index.nd_idx[i] * strides.nd_idx[i];
  }
  return nd_index;
}

template <int NDIM>
__device__ int64_t device_nd_to_flat(NdIndex<NDIM> &nd_index,
                                     const NdIndex<NDIM> strides) {
  auto idx = 0;
  for (int i = 0; i < NDIM; i++) {
    idx += strides.nd_idx[i] * nd_index.nd_idx[i];
  }
  return idx;
}

template <int NDIM, typename T>
__host__ NdIndex<NDIM> to_nd_index(const T &vec) {
  NdIndex<NDIM> nd_index;
  for (int i = 0; i < NDIM; i++) {
    nd_index.nd_idx[i] = vec[i];
  }
  return nd_index;
}
}

#endif
