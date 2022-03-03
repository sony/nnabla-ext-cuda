// Copyright 2021 Sony Corporation.
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

#ifndef __NBLA_CUDA_UTILS_STFT_CUH__
#define __NBLA_CUDA_UTILS_STFT_CUH__

#include <nbla/cuda/function/utils/stft.hpp>

template <typename T, stft::WINDOW_TYPE window_type>
__global__ void kernel_window(const int window_size, const int fft_size, T *w) {
  NBLA_CUDA_KERNEL_LOOP(idx, fft_size) {
    const auto left_pad = (fft_size - window_size) / 2;
    const auto w_idx = idx - left_pad;

    const auto r_window_size = 1.0f / window_size;
    if (0 <= w_idx && w_idx < window_size) {
      if (window_type == stft::WINDOW_TYPE::hanning) {
        w[idx] = (T)0.5 - (T)0.5 * cospif(2.0f * w_idx * r_window_size);
      } else if (window_type == stft::WINDOW_TYPE::hamming) {
        w[idx] = (T)0.54 - (T)0.46 * cospif(2.0f * w_idx * r_window_size);
      } else { // window_type == istft::WINDOW_TYPE::rectangular
        w[idx] = (T)1;
      }
    } else {
      w[idx] = (T)0;
    }
  }
}

#endif
