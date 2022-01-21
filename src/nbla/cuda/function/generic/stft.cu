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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/stft.hpp>
#include <nbla/cuda/function/utils/stft.cuh>
#include <nbla/variable.hpp>

#include <nbla/cuda/function/istft.hpp>

namespace nbla {

template <typename T>
void STFTCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  STFT<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  this->window_type_t_ = string_to_window_type(this->window_type_);

  if (this->as_istft_backward_) {
    // For the use of some ISTFTCuda methods.
    istft_cuda_ = make_shared<ISTFTCuda<T>>(
        this->ctx_, this->window_size_, this->stride_, this->fft_size_,
        this->window_type_, this->center_, this->pad_mode_,
        false /* as_stft_backward */);
    Variable dummy_y_r(outputs[0]->shape()), dummy_y_i(outputs[1]->shape()),
        dummy_x;
    istft_cuda_->setup({&dummy_y_r, &dummy_y_i}, {&dummy_x});
  }
}

template <typename T>
__global__ void kernel_conv_weight(const int fft_size, const int weight_size,
                                   const T *window, T *conv_r, T *conv_i) {
  NBLA_CUDA_KERNEL_LOOP(idx, weight_size) {
    const auto w_idx = idx / fft_size;
    const auto t_idx = idx - w_idx * fft_size; // idx % fft_size

    const auto r_fft_size = 1.0f / fft_size;
    T mat_r = cospif(2.0f * w_idx * t_idx * r_fft_size);
    T mat_i = -sinpif(2.0f * w_idx * t_idx * r_fft_size);

    conv_r[idx] = mat_r * window[t_idx];
    conv_i[idx] = mat_i * window[t_idx];
  }
}

template <typename T>
void STFTCuda<T>::calculate_conv_weight(Variable &conv_r, Variable &conv_i) {
  if (this->as_istft_backward_) {
    istft_cuda_->calculate_conv_weight(conv_r, conv_i);
    return;
  }

  // Create window
  auto w_ptr =
      this->window_.template cast_data_and_get_pointer<Tcu>(this->ctx_);

  using WINDOW_TYPE = stft::WINDOW_TYPE;
  constexpr auto hanning = WINDOW_TYPE::hanning;
  constexpr auto hamming = WINDOW_TYPE::hamming;
  constexpr auto rectangular = WINDOW_TYPE::rectangular;

  auto kernel = kernel_window<Tcu, rectangular>;
  if (window_type_t_ == hanning) {
    kernel = kernel_window<Tcu, hanning>;
  } else if (window_type_t_ == hamming) {
    kernel = kernel_window<Tcu, hamming>;
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->window_size_, this->fft_size_,
                                 w_ptr);

  // Calculate DFT (descrete Fourier transform) coefficients and merge with
  // window func.
  auto conv_r_ptr = conv_r.cast_data_and_get_pointer<Tcu>(this->ctx_);
  auto conv_i_ptr = conv_i.cast_data_and_get_pointer<Tcu>(this->ctx_);
  const auto w_data_ptr =
      this->window_.template get_data_pointer<Tcu>(this->ctx_);

  const int size = conv_r.size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_conv_weight<Tcu>, this->fft_size_, size,
                                 w_data_ptr, conv_r_ptr, conv_i_ptr);

  // Clear internal buffer
  this->window_.data()->array()->clear();
}

template <typename T>
void STFTCuda<T>::apply_inv_window_forward(Variable *x, Variable *y) {
  istft_cuda_->apply_inv_window_forward(x, y);
}

template <typename T>
void STFTCuda<T>::apply_inv_window_backward(Variable *x, Variable *y,
                                            const bool accum) {
  istft_cuda_->apply_inv_window_backward(x, y, accum);
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
