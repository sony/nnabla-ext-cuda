// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

#include <nbla/cuda/common.hpp>

namespace nbla {

__inline__ __device__ int transpose_offset(const int idx, const int ndim,
                                           const int *axes,
                                           const int *x_strides,
                                           const int *y_strides,
                                           const int *y_shape) {
  int i = 0;
  for (int d = 0; d < ndim; ++d) {
    const int k = int(idx / y_strides[d]) % y_shape[d];
    i += k * x_strides[axes[d]];
  }
  return i;
}

template <typename T>
__global__ void transpose_kernel(const int num, const int ndim, const int *axes,
                                 const int *x_strides, const int *y_strides,
                                 const int *y_shape, const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(o, num) {
    int i = transpose_offset(o, ndim, axes, x_strides, y_strides, y_shape);
    y[o] = x[i];
  }
}

template <typename T>
__global__ void transpose_2value_kernel(const int num, const int ndim,
                                        const int *axes, const int *x_strides,
                                        const int *y_strides,
                                        const int *y_shape, const T *in1,
                                        const T *in2, T *out1, T *out2) {
  NBLA_CUDA_KERNEL_LOOP(o, num) {
    int i = transpose_offset(o, ndim, axes, x_strides, y_strides, y_shape);
    out1[o] = in1[i];
    out2[o] = in2[i];
  }
}
}
