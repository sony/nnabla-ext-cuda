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
  const int n = inputs[0]->size() / inputs[0]->size(kernel_base);
  auto dtype = cudnn_data_type<T>::type();

  if (this->kernel_.size() == 2) {
    const int xh = inshape[kernel_base + 0];
    const int xw = inshape[kernel_base + 1];
    const int yh = outshape[kernel_base + 0];
    const int yw = outshape[kernel_base + 1];
    NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW,
                                                dtype, n, 1, xh, xw));
    NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW,
                                                dtype, n, 1, yh, yw));
  } else if (this->kernel_.size() == 3) {
    const int xd = inshape[kernel_base + 0];
    const int xh = inshape[kernel_base + 1];
    const int xw = inshape[kernel_base + 2];
    const int yd = outshape[kernel_base + 0];
    const int yh = outshape[kernel_base + 1];
    const int yw = outshape[kernel_base + 2];
    const int xshape[5] = {1, n, xd, xh, xw};
    const int yshape[5] = {1, n, yd, yh, yw};
    const int xstrides[5] = {n * xd * xh * xw, xd * xh * xw, xh * xw, xw, 1};
    const int ystrides[5] = {n * yd * yh * yw, yd * yh * yw, yh * yw, yw, 1};
    NBLA_CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(input_desc_, dtype, 5, xshape, xstrides));
    NBLA_CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(output_desc_, dtype, 5, yshape, ystrides));
  }
}

template <class T>
void AveragePoolingCudaCudnn<T>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
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
  Tw *dx = inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[0]);
  const Tw *dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  // *y and *x are  not used in NNabla, but they are required with cudnn API
  const Tw *y = outputs[0]->get_data_pointer<Tw>(this->ctx_);
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
  NBLA_CUDNN_CHECK(cudnnPoolingBackward(
      cudnn_handle_, pooling_desc_, &alpha, output_desc_, y, output_desc_, dy,
      input_desc_, x, &beta, input_desc_, dx));
}
}
