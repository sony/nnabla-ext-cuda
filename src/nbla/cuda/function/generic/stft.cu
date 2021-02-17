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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/stft.hpp>
#include <nbla/cuda/function/utils/stft.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void STFTCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  STFT<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  this->window_type_t_ = string_to_window_type(this->window_type_);
}

template <typename T>
__global__ void kernel_conv_weight(const int fft_size, const int weight_size,
                                   const T *window_func, T *conv_r, T *conv_i) {
  NBLA_CUDA_KERNEL_LOOP(idx, weight_size) {
    const auto w_idx = idx / fft_size;
    const auto t_idx = idx - w_idx * fft_size; // idx % fft_size

    const auto r_fft_size = 1.0f / fft_size;
    T mat_r = cospif(2.0f * w_idx * t_idx * r_fft_size);
    T mat_i = -sinpif(2.0f * w_idx * t_idx * r_fft_size);

    conv_r[idx] = mat_r * window_func[t_idx];
    conv_i[idx] = mat_i * window_func[t_idx];
  }
}

template <typename T>
void STFTCuda<T>::calculate_conv_weight(Variable &conv_r, Variable &conv_i) {
  // compute window func
  Variable window_func({this->fft_size_});
  auto wf_ptr = window_func.cast_data_and_get_pointer<Tcu>(this->ctx_);

  using WINDOW_TYPE = stft::WINDOW_TYPE;
  constexpr auto hanning = WINDOW_TYPE::hanning;
  constexpr auto hamming = WINDOW_TYPE::hamming;
  constexpr auto rectangular = WINDOW_TYPE::rectangular;

  if (window_type_t_ == hanning) {
    auto kernel = kernel_window_func<Tcu, hanning>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->window_size_, this->fft_size_,
                                   wf_ptr);
  } else if (window_type_t_ == hamming) {
    auto kernel = kernel_window_func<Tcu, hamming>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->window_size_, this->fft_size_,
                                   wf_ptr);
  } else {
    auto kernel = kernel_window_func<Tcu, rectangular>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->window_size_, this->fft_size_,
                                   wf_ptr);
  }

  // calculate STFT filter coefficients
  auto conv_r_ptr = conv_r.cast_data_and_get_pointer<Tcu>(this->ctx_);
  auto conv_i_ptr = conv_i.cast_data_and_get_pointer<Tcu>(this->ctx_);
  const auto wf_data_ptr = window_func.get_data_pointer<Tcu>(this->ctx_);

  const int size = conv_r.size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_conv_weight<Tcu>, this->fft_size_, size,
                                 wf_data_ptr, conv_r_ptr, conv_i_ptr);
}

template <typename T>
void STFTCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);
  STFT<T>::forward_impl(inputs, outputs);
}

template <typename T>
void STFTCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  STFT<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
