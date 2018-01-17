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

// sigmoid.cpp

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/sigmoid.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

template <typename T>
void SigmoidCudaCudnn<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, inputs[0]->size()));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, outputs[0]->size()));
}

template <class T>
void SigmoidCudaCudnn<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = 0;
#if CUDNN_VERSION >= 5000
  NBLA_CUDNN_CHECK(cudnnActivationForward(cudnn_handle_, activation_desc_,
                                          &alpha, input_desc_, x, &beta,
                                          output_desc_, y));
#else
  NBLA_CUDNN_CHECK(cudnnActivationForward_v4(cudnn_handle_, activation_desc_,
                                             &alpha, input_desc_, x, &beta,
                                             output_desc_, y));
#endif
}

template <class T>
void SigmoidCudaCudnn<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = accum[0] ? 1 : 0;
#if CUDNN_VERSION >= 5000
  NBLA_CUDNN_CHECK(cudnnActivationBackward(
      cudnn_handle_, activation_desc_, &alpha, output_desc_, y,
      this->output_desc_, dy, input_desc_, x, &beta, input_desc_, dx));
#else
  NBLA_CUDNN_CHECK(cudnnActivationBackward_v4(
      cudnn_handle_, activation_desc_, &alpha, output_desc_, y,
      this->output_desc_, dy, input_desc_, x, &beta, input_desc_, dx));
#endif
}
}
