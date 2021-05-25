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
#include <nbla/cuda/function/istft.hpp>
#include <nbla/cuda/function/kernel/istft.cuh>
#include <nbla/cuda/function/utils/stft.cuh>
#include <nbla/variable.hpp>

#include <nbla/cuda/function/stft.hpp>

namespace nbla {

template <typename T>
void ISTFTCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  this->window_type_t_ = string_to_window_type(this->window_type_);

  ISTFT<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (this->as_stft_backward_) {
    // For the use of some STFTCuda methods.
    stft_cuda_ = make_shared<STFTCuda<T>>(
        this->ctx_, this->window_size_, this->stride_, this->fft_size_,
        this->window_type_, this->center_, this->pad_mode_,
        false /* as_istft_backward */);
    Variable dummy_x(outputs[0]->shape()), dummy_y_r(inputs[0]->shape()),
        dummy_y_i(inputs[1]->shape());
    stft_cuda_->setup({&dummy_x}, {&dummy_y_r, &dummy_y_i});
  }
}

template <typename T>
void ISTFTCuda<T>::calculate_window(Context &ctx, Variable *window) const {
  using WINDOW_TYPE = stft::WINDOW_TYPE;
  constexpr auto hanning = WINDOW_TYPE::hanning;
  constexpr auto hamming = WINDOW_TYPE::hamming;
  constexpr auto rectangular = WINDOW_TYPE::rectangular;

  auto window_data = window->cast_data_and_get_pointer<Tcu>(ctx);

  auto kernel = kernel_window<Tcu, rectangular>;
  if (window_type_t_ == hanning) {
    kernel = kernel_window<Tcu, hanning>;
  } else if (window_type_t_ == hamming) {
    kernel = kernel_window<Tcu, hamming>;
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, this->window_size_, this->fft_size_,
                                 window_data);
}

template <typename T>
void ISTFTCuda<T>::calculate_inv_window(Context &ctx, Variable *inv_window) {
  // Create window
  calculate_window(ctx, &this->window_);

  // Calculate inv_window
  const int size = inv_window->size();
  const auto window_data = this->window_.template get_data_pointer<Tcu>(ctx);
  auto inv_window_data = inv_window->cast_data_and_get_pointer<Tcu>(ctx);
  auto kernel = kernel_inv_window<Tcu>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, this->fft_size_, this->stride_,
                                 window_data, inv_window_data);

  // Clear internal buffer
  this->window_.data()->array()->clear();
}

template <typename T>
void ISTFTCuda<T>::apply_inv_window_forward(Variable *x, Variable *y) {
  // Create inv_window
  calculate_inv_window(this->ctx_, &this->inv_window_);

  // Apply inv_window
  const int x_size = x->size();
  const int inv_window_size = this->inv_window_.size();
  const auto inv_window_data =
      this->inv_window_.template get_data_pointer<Tcu>(this->ctx_);
  const auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto kernel = this->center_ ? kernel_apply_inv_window_forward<Tcu, true>
                              : kernel_apply_inv_window_forward<Tcu, false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, x_size, inv_window_size, x_data,
                                 y_data, inv_window_data, this->fft_size_);

  // Clear internal buffer
  this->inv_window_.data()->array()->clear();
}

template <typename T>
void ISTFTCuda<T>::apply_inv_window_backward(Variable *x, Variable *y,
                                             const bool accum) {
  // Create inv_window
  calculate_inv_window(this->ctx_, &this->inv_window_);

  // Apply inv_window
  const int x_size = x->size();
  const int inv_window_size = this->inv_window_.size();
  const auto inv_window_data =
      this->inv_window_.template get_data_pointer<Tcu>(this->ctx_);
  auto x_grad = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum);
  const auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);
  const int pad_size = this->fft_size_ / 2;

  auto kernel = kernel_apply_inv_window_backward<Tcu, true /* center */,
                                                 true /* accum */>;
  if (this->center_) {
    kernel = accum ? kernel_apply_inv_window_backward<Tcu, true, true>
                   : kernel_apply_inv_window_backward<Tcu, true, false>;
  } else {
    kernel = accum ? kernel_apply_inv_window_backward<Tcu, false, true>
                   : kernel_apply_inv_window_backward<Tcu, false, false>;
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, x_size, inv_window_size, pad_size,
                                 x_grad, y_grad, inv_window_data);

  // Clear internal buffer
  this->inv_window_.data()->array()->clear();
}

template <typename T>
void ISTFTCuda<T>::calculate_conv_weight(Variable &conv_cos,
                                         Variable &conv_sin) {
  if (this->as_stft_backward_) {
    stft_cuda_->calculate_conv_weight(conv_cos, conv_sin);
    return;
  }

  // Compute window
  calculate_window(this->ctx_, &this->window_);

  // Calculate IDFT (inverse descrete Fourier transform) coefficients and merge
  // with window func.
  auto conv_cos_ptr = conv_cos.cast_data_and_get_pointer<Tcu>(this->ctx_);
  auto conv_sin_ptr = conv_sin.cast_data_and_get_pointer<Tcu>(this->ctx_);
  const auto w_data_ptr =
      this->window_.template get_data_pointer<Tcu>(this->ctx_);

  const int size = conv_cos.size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_conv_weight<Tcu>, this->fft_size_,
                                 this->stride_, size, w_data_ptr, conv_cos_ptr,
                                 conv_sin_ptr);

  // Clear internal buffer
  this->window_.data()->array()->clear();
}

template <typename T>
void ISTFTCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(this->device_);
  ISTFT<T>::forward_impl(inputs, outputs);
}

template <typename T>
void ISTFTCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);
  ISTFT<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
