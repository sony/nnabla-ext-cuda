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

// Note:
// __inline__ qualifier must be needed to avoid multiple function definition in
// linking,
// though compilar warn like "inline qualifier ignored for "__global__"
// function".
// We should ignore this compilar warning.

#ifndef __NBLA_CUDA_UTILS_RELU_CUH__
#define __NBLA_CUDA_UTILS_RELU_CUH__

#include <algorithm>
#include <nbla/cuda/common.hpp>

namespace nbla {

template <class T> inline Size_t interpret_size(const Size_t size) {
  return size; // interpreted as half type
}

template <> inline Size_t interpret_size<Half>(const Size_t size) {
  return (size + 1) / 2; // interpreted as half2 type
}

template <> inline Size_t interpret_size<HalfCuda>(const Size_t size) {
  return (size + 1) / 2; // interpreted as half2 type
}

//=============================================================================
// General implementaion
//=============================================================================
// They have the same performance as cuDNN 8.1.1 for float)
template <typename T>
__inline__ __global__ void
kernel_relu_forward(const Size_t size2, const Size_t size, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size) { y[idx] = max((T)0, x[idx]); }
}

template <bool accum, typename T>
__inline__ __global__ void kernel_relu_backward(const Size_t size2,
                                                const Size_t size, T *dx,
                                                const T *y, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size) {
    dx[idx] = (accum ? dx[idx] : (T)0) + (y[idx] > (T)0 ? dy[idx] : (T)0);
  }
}

//=============================================================================
// Vectrized implementation with half2
//=============================================================================
// Optimizing the number of trasactions for global memory access
// by using half2 type.
__inline__ __device__ void kernel_relu_forward_half2(const Size_t size2,
                                                     const Size_t size,
                                                     HalfCuda *y,
                                                     const HalfCuda *x) {
  // HalfCuda is aligned 2. See nbla/cuda/half.hpp.
  half2 *y2 = reinterpret_cast<half2 *>(y);
  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  const HalfCuda zero(0);

  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx2, size2) {
    const Size_t idx = 2 * idx2;
    if (idx + 1 == size) {
      // The last fraction element cannot be vectrized into half2.
      y[idx] = max(zero, x[idx]);
    } else {
      // 1. Load the two elements once.
      const half2 x2_val = x2[idx2];
      HalfCuda y_buff[2];

      // 2. Compute ReLU respectively.
      const HalfCuda low(__low2half(x2_val));
      const HalfCuda high(__high2half(x2_val));
      y_buff[0] = (low > zero ? low : zero);
      y_buff[1] = (high > zero ? high : zero);
      // Note: __hmax2 is not supported before CUDA 11.1. Here we do not
      //       use the macro switch to it because it did not improve
      //       the speed.

      // 3. Store the two elements once.
      y2[idx2] = __halves2half2(y_buff[0].h, y_buff[1].h);
    }
  }
}

template <bool accum>
__inline__ __device__ void
kernel_relu_backward_half2(const Size_t size2, const Size_t size, HalfCuda *dx,
                           const HalfCuda *y, const HalfCuda *dy) {
  // HalfCuda is aligned 2. See nbla/cuda/half.hpp.
  half2 *dx2 = reinterpret_cast<half2 *>(dx);
  const half2 *y2 = reinterpret_cast<const half2 *>(y);
  const half2 *dy2 = reinterpret_cast<const half2 *>(dy);
  const HalfCuda zero(0);

  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx2, size2) {
    const Size_t idx = 2 * idx2;
    if (idx + 1 == size) {
      // The last fraction element cannot be vectrized into half2.
      dx[idx] = (accum ? dx[idx] : zero) + (y[idx] > zero ? dy[idx] : zero);
    } else {
      // 1. Load the two elements once.
      const half2 y2_val = y2[idx2];
      const half2 dy2_val = dy2[idx2];
      HalfCuda dx_buff[2];

      // 2. Compute the gradient of ReLU respectively.
      dx_buff[0] =
          (HalfCuda(__low2half(y2_val)) > zero ? HalfCuda(__low2half(dy2_val))
                                               : zero);
      dx_buff[1] =
          (HalfCuda(__high2half(y2_val)) > zero ? HalfCuda(__high2half(dy2_val))
                                                : zero);

      // 3. Store the two elements once.
      if (accum) {
        // half2 arithmetic is not supported in lower compute capability.
        const half2 dx_tmp = dx2[idx2];
        const HalfCuda low = dx_buff[0] + HalfCuda(__low2half(dx_tmp));
        const HalfCuda high = dx_buff[1] + HalfCuda(__high2half(dx_tmp));
        dx2[idx2] = __halves2half2(low.h, high.h);
      } else {
        dx2[idx2] = __halves2half2(dx_buff[0].h, dx_buff[1].h);
      }
    }
  }
}

// C++11 cannot specialize tmplate functions partially.
template <>
__inline__ __global__ void kernel_relu_forward(const Size_t size2,
                                               const Size_t size, HalfCuda *y,
                                               const HalfCuda *x) {
  kernel_relu_forward_half2(size2, size, y, x);
}

template <>
__inline__ __global__ void
kernel_relu_backward<true>(const Size_t size2, const Size_t size, HalfCuda *dx,
                           const HalfCuda *y, const HalfCuda *dy) {
  kernel_relu_backward_half2<true>(size2, size, dx, y, dy);
}

template <>
__inline__ __global__ void
kernel_relu_backward<false>(const Size_t size2, const Size_t size, HalfCuda *dx,
                            const HalfCuda *y, const HalfCuda *dy) {
  kernel_relu_backward_half2<false>(size2, size, dx, y, dy);
}
}

#endif