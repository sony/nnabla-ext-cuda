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

#ifndef __NBLA_CUDA_UTILS_INDEX_CONVERTER_CUH__
#define __NBLA_CUDA_UTILS_INDEX_CONVERTER_CUH__

namespace nbla {

// The size of fixsed length array restricts the number of array dimensions.
// There is no significant reason of the value 16.
#define NBLA_CUDA_MAX_NDIM 16

/** This class stores a pair of quotient and remainder resulted by a division.
 */
template <class T> struct DivMod {
  T quotient, remainder;
  __device__ DivMod(const T q, const T r) : quotient(q), remainder(r) {}
};

/** This class provides the CUDA kernel to perform the division
    with a constant divisor.
 */
template <class T> class Divider {
  T divisor_;

public:
  Divider() {}
  Divider(const T divisor) : divisor_(divisor) {}

  __device__ DivMod<T> divmod(const T dividend) const {
    T q = dividend / divisor_;
    return DivMod<T>(q, dividend - q * divisor_);
  }
};

/** This specified class Divider<uint32_t> provides the optimized CUDA kernel
    to perform the fast division with a constant divisor by the algorithm [1].

    [1] Division by Invariant Integers Using Multiplication,
        Torbjörn Granlund and Peter L. Montgomery, 1994,
        https://doi.org/10.1145/773473.178249.
 */
template <> class Divider<uint32_t> {
  uint32_t divisor_;
  uint32_t magic_;
  uint32_t shift_;

public:
  Divider() {}
  Divider(const uint32_t divisor) : divisor_(divisor) {
    shift_ = 0;
    for (; shift_ < 32; shift_++) {
      if ((1U << shift_) >= divisor_) {
        break;
      }
    }
    uint64_t one = 1;
    magic_ = ((one << 32) * ((one << shift_) - divisor_)) / divisor_ + 1;
  }

  __device__ DivMod<uint32_t> divmod(const uint32_t dividend) const {
    uint32_t t = __umulhi(dividend, magic_);
    uint32_t div = (t + dividend) >> shift_;
    return DivMod<uint32_t>(div, dividend - div * divisor_);
  }
};

/** This utility class defines the useful protocol to convert an index
    which is typically used in CUDA kernels.

    - change_strides: decomposing the given linear index in vector component
                      form by using "strides" (given by the class constructor)
                      and then composing the new linear index from them by using
                      "other_strides" (also given by the class constructor).
 */
template <class IndexT> class IndexConverter {
  int ndim_;
  Divider<IndexT> divider_[NBLA_CUDA_MAX_NDIM];
  IndexT other_strides_[NBLA_CUDA_MAX_NDIM];

public:
  IndexConverter(const Shape_t &strides, const Shape_t &other_strides)
      : ndim_(strides.size()) {
    NBLA_CHECK(strides.size() == other_strides.size(), error_code::value,
               "All sizes for IndexConverter must be same.");
    NBLA_CHECK(ndim_ <= NBLA_CUDA_MAX_NDIM, error_code::value,
               "The size for IndexConverter must be <= %d.",
               NBLA_CUDA_MAX_NDIM);

    for (int i = 0; i < ndim_; i++) {
      divider_[i] = Divider<IndexT>(strides[i]);
      other_strides_[i] = other_strides[i];
    }
  }

  /** Decomposing the given linear index in vector component form by using
      "strides" (given by the class constructor) and then composing the new
      linear index from them by using "other_strides" (also given by the class
      constructor). The new index is returned adding the base index.
   */
  __device__ IndexT change_strides(IndexT base_idx, IndexT idx) const {
#pragma unroll
    for (int i = 0; i < NBLA_CUDA_MAX_NDIM; i++) {
      if (i == ndim_) {
        break;
      }
      auto divmod = divider_[i].divmod(idx);
      idx = divmod.remainder;
      base_idx += divmod.quotient * other_strides_[i];
    }

    return base_idx;
  }
};
}
#endif