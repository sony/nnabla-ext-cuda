// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_CUDA_UTILS_TYPES_CUH__
#define __NBLA_CUDA_UTILS_TYPES_CUH__

namespace nbla {
struct floatint {
  float f;
  int i;
};

template <typename IndexT> struct WelfordType {
  float mean;
  float m2;
  IndexT n;
};

template <class T, class IndexT> struct ValWithIdx {
  T val;
  IndexT idx;
  __device__ ValWithIdx(const T v, const IndexT i) : val(v), idx(i) {}
  __device__ ValWithIdx() {}
};

//==============================================================================
// Type sets for reduction CUDA kernel
//==============================================================================
// Tcu, IndexT, and StorageT are reuired as the interface of type set.
template <class T, class U> struct ReduceOpSumLikeType {
  using Tcu = typename CudaType<T>::type;
  using IndexT = U;
  using StorageT = typename CudaTypeForceFloat<T>::type;
};

template <class T, class U> struct ReduceOpMaxLikeType {
  using Tcu = typename CudaType<T>::type;
  using IndexT = U;
  using StorageT = ValWithIdx<Tcu, IndexT>;
};

//==============================================================================
// Type sets for scan CUDA kernel
//==============================================================================
// Tcu, IndexT, and StorageT are reuired as the interface of type set.
// Type for scan sum

/**
 * @brief Type for cumulative sum scan.
 *
 * @tparam T the type of the input and output values;
 * @tparam U the type of the size, shape, and indices of the input and output.
 */
template <class T, class U> struct ScanOpSumType {
  using Tcu = typename CudaType<T>::type;
  using IndexT = U;
  using StorageT = typename CudaTypeForceFloat<T>::type;
};

/**
 * @brief Type for cumulative prod scan.
 *
 * @tparam T the type of the input and output values;
 * @tparam U the type of the size, shape, and indices of the input and output.
 */
template <class T, class U> using ScanOpProdType = ScanOpSumType<T, U>;
}

#endif
