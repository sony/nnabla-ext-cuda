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

template <typename T>
__global__ void
im2col_kernel(const int n, const T *img, const int height, const int width,
              const int kernel_h, const int kernel_w, const int pad_h,
              const int pad_w, const int stride_h, const int stride_w,
              const int dilation_h, const int dilation_w, const int h_o,
              const int w_o, T *col);

template <typename T>
void im2col_cuda(const T *img, const int c_i, const int *shape, const int *k,
                 const int *p, const int *s, const int *d, T *col) {
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int col_size = c_i * k[0] * k[1] * h_o * w_o;
  im2col_kernel<T><<<NBLA_CUDA_GET_BLOCKS(col_size), NBLA_CUDA_NUM_THREADS>>>(
      col_size, img, shape[0], shape[1], k[0], k[1], p[0], p[1], s[0], s[1],
      d[0], d[1], h_o, w_o, col);
}

template <typename T>
void im2col_nd_cuda(const T *img, const int c, const int spatial_dims,
                    const int *spatial_shape, const int *kernel, const int *pad,
                    const int *stride, const int *dilation, T *col) {
  NBLA_ERROR(error_code::not_implemented, "Im2Col_ND is not implemented.");
}

template <typename T>
__global__ void
im2col_kernel(const int col_size, const T *img, const int height,
              const int width, const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w, const int stride_h,
              const int stride_w, const int dilation_h, const int dilation_w,
              const int h_o, const int w_o, T *col) {
  NBLA_CUDA_KERNEL_LOOP(idx, col_size) {
    const int c_col = idx / (kernel_h * kernel_w * h_o * w_o);
    const int h_idx = idx / w_o;
    const int h_col = h_idx % h_o;
    const int w_col = idx % w_o;
    const int c_im = h_idx / h_o;
    const int y_k = c_im / kernel_w % kernel_h;
    const int x_k = c_im % kernel_w;
    const int y_i = h_col * stride_h - pad_h + y_k * dilation_h;
    const int x_i = w_col * stride_w - pad_w + x_k * dilation_w;
    col[idx] = (y_i >= 0 && x_i >= 0 && y_i < height && x_i < width)
#if __CUDA_ARCH__ >= 350
                   ? __ldg(&img[(c_col * height + y_i) * width + x_i])
                   : 0;
#else
                   ? img[(c_col * height + y_i) * width + x_i]
                   : 0;
#endif
  }
}
template <>
inline __global__ void
im2col_kernel(const int col_size, const HalfCuda *img, const int height,
              const int width, const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w, const int stride_h,
              const int stride_w, const int dilation_h, const int dilation_w,
              const int h_o, const int w_o, HalfCuda *col) {
  NBLA_CUDA_KERNEL_LOOP(idx, col_size) {
    const int c_col = idx / (kernel_h * kernel_w * h_o * w_o);
    const int h_idx = idx / w_o;
    const int h_col = h_idx % h_o;
    const int w_col = idx % w_o;
    const int c_im = h_idx / h_o;
    const int y_k = c_im / kernel_w % kernel_h;
    const int x_k = c_im % kernel_w;
    const int y_i = h_col * stride_h - pad_h + y_k * dilation_h;
    const int x_i = w_col * stride_w - pad_w + x_k * dilation_w;
    col[idx] = (y_i >= 0 && x_i >= 0 && y_i < height && x_i < width)
                   ? img[(c_col * height + y_i) * width + x_i]
                   : (HalfCuda)0;
  }
}
}
