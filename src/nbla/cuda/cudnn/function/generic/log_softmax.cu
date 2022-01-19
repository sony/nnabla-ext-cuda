// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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

// log_softmax.cu

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/log_softmax.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void LogSoftmaxCudaCudnn<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  LogSoftmax<T>::setup_impl(inputs, outputs);
  auto dtype = cudnn_data_type<T>::type();
  cudnn_softmax_ = CudnnSoftmax::create(
      inputs[0]->shape(), this->axis_, CUDNN_SOFTMAX_LOG, dtype, this->device_);
}

template <class T>
void LogSoftmaxCudaCudnn<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  NBLA_CHECK(cudnn_softmax_, error_code::value, "setup not called.");
  auto x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  cudnn_softmax_->forward(&alpha, x, &beta, y);
}

template <class T>
void LogSoftmaxCudaCudnn<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  NBLA_CHECK(cudnn_softmax_, error_code::value, "setup not called.");
  auto y = outputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  auto dx = inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[0]);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
  cudnn_softmax_->backward(&alpha, y, dy, &beta, dx);
}
} // namespace nbla
