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

// AveragePoolingCudaCudnn.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/cudnn/function/average_pooling.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

template <typename T>
void AveragePoolingCudaCudnn<T>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {

  AveragePooling<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  Shape_t inshape = inputs[0]->shape();
  Shape_t outshape = outputs[0]->shape();
  const int kernel_base = inshape.size() - this->kernel_.size();
  const int hx = inshape[kernel_base + 0];
  const int wx = inshape[kernel_base + 1];
  const int hy = outshape[kernel_base + 0];
  const int wy = outshape[kernel_base + 1];
  const int n_map = inputs[0]->size() / inputs[0]->size(kernel_base);
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), n_map,
                                              1, hx, wx));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), n_map,
                                              1, hy, wy));
}

template <class T>
void AveragePoolingCudaCudnn<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = 0;
  NBLA_CUDNN_CHECK(cudnnPoolingForward(cudnn_handle_, pooling_desc_, &alpha,
                                       input_desc_, x, &beta, output_desc_, y));
}

template <class T>
void AveragePoolingCudaCudnn<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  // *y and *x are  not used in NNabla, but they are required with cudnn API
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = (accum[0] ? 1 : 0);
  NBLA_CUDNN_CHECK(cudnnPoolingBackward(
      cudnn_handle_, pooling_desc_, &alpha, output_desc_, y, output_desc_, dy,
      input_desc_, x, &beta, input_desc_, dx));
}

// Template instanciation
template class AveragePoolingCudaCudnn<float>;
}
