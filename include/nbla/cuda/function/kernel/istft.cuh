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

// All of the following CUDA kernels and constants are copied from PyTorch (link
// below) and modified/optimized for NNabla.
// https://github.com/pytorch/pytorch/blob/32b37ba2462d9d87337a4fe332f95524a4c49777/aten/src/ATen/native/cuda/Normalization.cuh

#include <nbla/cuda/common.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_inv_window(const int size, const int fft_size,
                                  const int stride, const T *window,
                                  T *inv_window) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    inv_window[idx] = T(0);

    for (int i = 0; i < fft_size / stride; i++) {
      const int w_idx = (idx + fft_size - i * stride) % fft_size;
      const bool right = idx - w_idx >= 0;
      const bool left = idx + (fft_size - w_idx - 1) < size;
      if (right && left) {
        const T w = window[w_idx];
        inv_window[idx] += w * w;
      }
    }
  }
}

template <typename T, bool center>
__global__ void
kernel_apply_inv_window_forward(const int x_size, const int inv_window_size,
                                const T *x, T *y, const T *inv_window,
                                const int fft_size) {
  NBLA_CUDA_KERNEL_LOOP(idx, x_size) {
    if (center) {
      if (fft_size / 2 <= idx % inv_window_size &&
          idx % inv_window_size < inv_window_size - fft_size / 2) {
        y[idx] = x[idx] / inv_window[idx % inv_window_size];
      }
    } else {
      y[idx] = x[idx] / inv_window[idx % inv_window_size];
    }
  }
}

template <typename T, bool center, bool accum>
__global__ void
kernel_apply_inv_window_backward(const int x_size, const int inv_window_size,
                                 const int pad_size, T *gx, const T *gy,
                                 const T *inv_window) {
  NBLA_CUDA_KERNEL_LOOP(idx, x_size) {
    const int iw_idx = idx % inv_window_size;
    if (center) {
      if (iw_idx < pad_size || inv_window_size - pad_size <= iw_idx) {
        // Avoid division by zero for padding region.
        gx[idx] = T(0);
      } else {
        gx[idx] = gy[idx] / inv_window[iw_idx] + (accum ? gx[idx] : (T)0);
      }
    } else {
      gx[idx] = gy[idx] / inv_window[iw_idx] + (accum ? gx[idx] : (T)0);
    }
  }
}

template <typename T>
__global__ void kernel_conv_weight(const int fft_size, const int stride,
                                   const int weight_size, const T *window,
                                   T *conv_r, T *conv_i) {
  NBLA_CUDA_KERNEL_LOOP(idx, weight_size) {
    const auto w_idx = idx / fft_size;
    const auto t_idx = idx - w_idx * fft_size; // idx % fft_size

    // Window
    const T w = window[t_idx];

    // Calculate inverse STFT filter coefficients
    const auto r_fft_size = 1.0f / fft_size;
    const auto alpha =
        (w_idx == 0 || w_idx == fft_size / 2 ? (T)1.0 : (T)2.0) * r_fft_size;
    const auto mat_cos = alpha * cospif(2.0f * w_idx * t_idx * r_fft_size);
    const auto mat_sin = alpha * -sinpif(2.0f * w_idx * t_idx * r_fft_size);

    conv_r[idx] = mat_cos * w;
    conv_i[idx] = mat_sin * w;
  }
}
}
