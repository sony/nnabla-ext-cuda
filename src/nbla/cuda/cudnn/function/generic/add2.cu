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

/** Add2
 */

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/add2.hpp>
#include <nbla/cuda/function/bc_add2.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <memory>

namespace nbla {

template <typename T>
void Add2CudaCudnn<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  if (inputs[0]->shape() != inputs[1]->shape()) {
    // Trying to fallback to broadcastable Add2.
    this->fall_back_func_ = std::shared_ptr<Function>(
        new BcAdd2Cuda<T>(this->ctx_, this->inplace_));
    this->fall_back_func_->setup(inputs, outputs);
    return;
  }

  Add2<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, inputs[0]->size()));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW,
                                              cudnn_data_type<T>::type(), 1, 1,
                                              1, outputs[0]->size()));
}

template <typename T>
void Add2CudaCudnn<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tw *x0 = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  const Tw *x1 = inputs[1]->get_data_pointer<Tw>(this->ctx_);
  Tw *y =
      outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, !this->inplace_);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(1);
  if (x0 == y) {
#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, x1,
                                    &beta, output_desc_, y));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, x1, &beta,
                                    output_desc_, y));
#endif
  } else if (x1 == y) {
#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, x0,
                                    &beta, output_desc_, y));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, x0, &beta,
                                    output_desc_, y));
#endif
  } else {
    Add2Cuda<T>::forward_impl(inputs, outputs);
  }
}

template <typename T>
void Add2CudaCudnn<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Tw *dx0 = inputs[0]->cast_grad_and_get_pointer<Tw>(
      this->ctx_, !(this->inplace_ || accum[0]));
  Tw *dx1 = inputs[1]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[1]);
  const Tw *dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  auto alpha = get_cudnn_scalar_arg<T>(1);

  if (dx0 != dy && propagate_down[0]) {
    auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, dy,
                                    &beta, output_desc_, dx0));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, dy, &beta,
                                    output_desc_, dx0));
#endif
  }
  // dx1 == dy never happens actually.
  if (dx1 != dy && propagate_down[1]) {
    auto beta = get_cudnn_scalar_arg<T>(accum[1] ? 1 : 0);
#if CUDNN_VERSION >= 4000
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, input_desc_, dy,
                                    &beta, output_desc_, dx1));
#else
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, CUDNN_ADD_FULL_TENSOR,
                                    &alpha, input_desc_, dy, &beta,
                                    output_desc_, dx1));
#endif
  }
}
}
