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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/ifft.hpp>
#include <nbla/cuda/function/utils/fft.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T> IFFTCuda<T>::~IFFTCuda() {
  cufftDestroy(plan_forward_);
  cufftDestroy(plan_backward_);
}

template <typename T>
void IFFTCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  cuda_set_device(this->device_);
  IFFT<T>::setup_impl(inputs, outputs);
  cufftCreate(&plan_forward_);
  cufftCreate(&plan_backward_);

  // Compute scale and store the original shape (i.e, n)
  Shape_t oshape(outputs[0]->shape());
  Size_t base_axis_output = oshape.size() - 1 - this->signal_ndim_;
  for (int i = 0; i < this->signal_ndim_; i++) {
    signal_size_ *= oshape[base_axis_output + i];
    n_.push_back(oshape[base_axis_output + i]);
  }
}

template <typename T>
void IFFTCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);

  // IFFT
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  exec_cufft<Tcu>(this->ctx_, x, y, inputs[0]->shape(), outputs[0]->shape(),
                  plan_forward_, true, true, CUFFT_INVERSE, this->n_,
                  this->signal_ndim_);

  // Normalize
  const Size_t size = outputs[0]->size();
  if (this->normalized_) {
    const float scale = 1.f / std::sqrt(this->signal_size_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                   y);
  } else {
    const float scale = 1.f / this->signal_size_;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                   y);
  }
}

template <typename T>
void IFFTCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  const Size_t size = inputs[0]->size();

  if (accum[0]) {
    // Create tmp array
    NdArrayPtr ndarray = make_shared<NdArray>(inputs[0]->shape());

    // FFT
    const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
    Tcu *tmp_buff = ndarray->cast(get_dtype<Tcu>(), this->ctx_)->pointer<Tcu>();
    exec_cufft<Tcu>(this->ctx_, dy, tmp_buff, outputs[0]->shape(),
                    inputs[0]->shape(), plan_backward_, true, true,
                    CUFFT_FORWARD, this->n_, this->signal_ndim_);

    // Normalize
    const Size_t size = inputs[0]->size();
    if (this->normalized_) {
      const float scale = 1.f / std::sqrt(this->signal_size_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                     tmp_buff);
    } else {
      const float scale = 1.f / this->signal_size_;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                     tmp_buff);
    }

    // Accumulation
    Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_cufft_result, size, tmp_buff, dx);

  } else {
    // FFT
    const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
    Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    exec_cufft<Tcu>(this->ctx_, dy, dx, outputs[0]->shape(), inputs[0]->shape(),
                    plan_backward_, true, true, CUFFT_FORWARD, this->n_,
                    this->signal_ndim_);
    // Normalize
    const Size_t size = inputs[0]->size();
    if (this->normalized_) {
      const float scale = 1.f / std::sqrt(this->signal_size_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                     dx);
    } else {
      const float scale = 1.f / this->signal_size_;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_normalize_cufft_result, size, scale,
                                     dx);
    }
  }
}
}
