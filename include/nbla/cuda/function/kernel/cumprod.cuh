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

#include <nbla/cuda/utils/atomic_min.cuh>

namespace nbla {

/**
 * This implementation is parallelized version of C++ implementation.
 * Its algorithm is exactly same as C++ one.
 * See details in `nnabla/src/nbla/function/generic/cumprod.cpp`.
 *
 * `i0` and `i2` in C++ code are parallelized here.
 */
template <typename T, typename AccumType, typename IndexT, bool accum,
          bool exclusive, bool reverse>
__global__ void
kernel_cumprod_backward(const IndexT size_outer_inner, const IndexT size_scan,
                        const IndexT size_inner, const T *x, const T *g_y,
                        AccumType *masked_cumprod, T *g_x) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size_outer_inner) {
    const IndexT i0 = idx / size_inner;
    const IndexT i2 = idx % size_inner;

    const IndexT offset = i0 * size_scan * size_inner + i2;

    // Create masked_cumprod
    IndexT first_zero_pos = size_scan;
    AccumType prod = (AccumType)1;
    for (IndexT k = 0; k < size_scan; k++) {
      const IndexT i1 = reverse ? size_scan - k - 1 : k;
      const IndexT idx = i1 * size_inner + offset;
      if (x[idx] == (T)0 && first_zero_pos == size_scan) {
        first_zero_pos = k;
        // prod *= (AccumType)1;
      } else {
        prod *= x[idx];
      }
      masked_cumprod[k * size_inner + offset] = prod;
    }

    // Calculate gradient
    AccumType sum = 0;
    for (IndexT k = size_scan - 1; k >= 0; k--) {
      const IndexT i1 = reverse ? size_scan - k - 1 : k;
      const IndexT idx = i1 * size_inner + offset;

      if (!exclusive) {
        sum += masked_cumprod[k * size_inner + offset] * g_y[idx];
      }

      T grad;
      if (k == first_zero_pos) {
        grad = (T)sum;
        sum = 0;
      } else if (k > first_zero_pos) {
        grad = (T)0;
      } else {
        grad = (T)sum / x[idx];
      }
      g_x[idx] = grad + (accum ? g_x[idx] : (T)0);

      if (exclusive && k != 0) {
        sum += masked_cumprod[(k - 1) * size_inner + offset] * g_y[idx];
      }
    }
  }
}

template <typename IndexT>
__global__ void kernel_fill_size_scan(const IndexT size_outer_inner,
                                      const IndexT size_scan,
                                      IndexT *first_zero_index) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size_outer_inner) {
    first_zero_index[idx] = size_scan;
  }
}

template <typename T, typename IndexT, bool reverse>
__global__ void kernel_first_zero_index(const IndexT size_input,
                                        const IndexT size_scan,
                                        const IndexT size_inner, const T *x,
                                        IndexT *first_zero_index) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size_input) {
    const IndexT i0 = idx / size_inner / size_scan;
    const IndexT i1 = idx / size_inner % size_scan;
    const IndexT i2 = idx % size_inner;
    const IndexT k = reverse ? size_scan - i1 - 1 : i1;
    if (x[idx] == (T)0) {
      atomic_min(&first_zero_index[i0 * size_inner + i2], k);
    }
  }
}

template <typename T, typename IndexT, bool reverse>
__global__ void kernel_mask_input(const IndexT size_input,
                                  const IndexT size_scan,
                                  const IndexT size_inner, const T *x,
                                  const IndexT *first_zero_index, T *y) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size_input) {
    const IndexT i0 = idx / size_inner / size_scan;
    const IndexT i1 = idx / size_inner % size_scan;
    const IndexT i2 = idx % size_inner;
    IndexT z_pos = first_zero_index[i0 * size_inner + i2];
    const IndexT k = reverse ? size_scan - i1 - 1 : i1;
    if (k == z_pos) {
      y[idx] = (T)1;
    } else {
      y[idx] = x[idx];
    }
  }
}

template <typename T, typename IndexT>
__global__ void kernel_prod_dy(const IndexT size, const T *cumprod,
                               const T *masked_cumprod, const T *dy,
                               T *out_cumprod, T *out_masked_cumprod) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size) {
    out_cumprod[idx] = cumprod[idx] * dy[idx];
    out_masked_cumprod[idx] = masked_cumprod[idx] * dy[idx];
  }
}

template <typename T, typename IndexT, bool reverse, bool accum>
__global__ void kernel_dx(const IndexT size_input, const IndexT size_scan,
                          const IndexT size_inner, const T *x, const T *cumsum,
                          const T *masked_cumsum,
                          const IndexT *first_zero_index, T *dx) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size_input) {
    const IndexT i0 = idx / size_inner / size_scan;
    const IndexT i1 = idx / size_inner % size_scan;
    const IndexT i2 = idx % size_inner;
    const IndexT z_pos = first_zero_index[i0 * size_inner + i2];
    const IndexT k = reverse ? size_scan - i1 - 1 : i1;
    T grad;
    if (k < z_pos) {
      grad = cumsum[idx] / x[idx];
    }
    if (k == z_pos) {
      grad = masked_cumsum[idx];
    }
    if (k > z_pos) {
      grad = (T)0;
    }
    dx[idx] = grad + (accum ? dx[idx] : (T)0);
  }
}
}
