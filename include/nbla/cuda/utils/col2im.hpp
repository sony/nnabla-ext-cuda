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
col2im_kernel(const int n, const T *col, const int height, const int width,
              const int channels, const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w, const int stride_h,
              const int stride_w, const int dilation_h, const int dilation_w,
              const int height_col, const int width_col, T *img);

template <typename T>
void col2im_cuda(const T *col, const int c_i, const int *shape, const int *k,
                 const int *p, const int *s, const int *d, T *img) {
  const int h_o = (shape[0] + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) / s[0] + 1;
  const int w_o = (shape[1] + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) / s[1] + 1;
  const int img_size = c_i * k[0] * k[1];
  col2im_kernel<T><<<NBLA_CUDA_GET_BLOCKS(img_size), NBLA_CUDA_NUM_THREADS>>>(
      img_size, col, shape[0], shape[1], c_i, k[0], k[1], p[0], p[1], s[0],
      s[1], d[0], d[1], h_o, w_o, img);
}

template <typename T>
void col2im_nd_cuda(const T *col, const int c, const int spatial_dims,
                    const int *spatial_shape, const int *kernel, const int *pad,
                    const int *stride, const int *dilation, T *img) {
  NBLA_ERROR(error_code::not_implemented, "Col2Im_ND is not implemented.");
}

// There's room for improvement
template <typename T>
__global__ void
col2im_kernel(const int ckhkw, const T *col, const int height, const int width,
              const int channels, const int kernel_h, const int kernel_w,
              const int pad_h, const int pad_w, const int stride_h,
              const int stride_w, const int dilation_h, const int dilation_w,
              const int h_o, const int w_o, T *img) {
  NBLA_CUDA_KERNEL_LOOP(cc, ckhkw) {
    const int x_k = cc % kernel_w;
    const int y_k = (cc / kernel_w) % kernel_h;
    const int z_k = cc / kernel_h / kernel_w;
    // Output image loop (columns of col matrix)
    for (int y_o = 0; y_o < h_o; ++y_o) {
      const int col_offset = (cc * h_o + y_o) * w_o;
      const int y_i =
          y_o * stride_h - pad_h + y_k * dilation_h; // y-pos in image
      // A technique for checking border with one comparison.
      if (static_cast<unsigned>(y_i) < static_cast<unsigned>(height)) {
        const int im_offset = (z_k * height + y_i) * width;
        for (int x_o = 0; x_o < w_o; ++x_o) {
          const int x_i =
              x_o * stride_w - pad_w + x_k * dilation_w; // x-pos in image
          if (static_cast<unsigned>(x_i) < static_cast<unsigned>(width)) {
            img[im_offset + x_i] += col[col_offset + x_o];
          }
        }
      }
    }
  }
}
}
