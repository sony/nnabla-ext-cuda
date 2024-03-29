// Copyright 2019,2020,2021 Sony Corporation.
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

#include <nbla/cuda/cudnn/function/utils/base_pooling.hpp>

namespace nbla {

template <typename BasePoolingType>
void BasePoolingCudaCudnn<BasePoolingType>::setup_impl(
    const Variables &inputs, const Variables &outputs) {
  NBLA_THIS_TYPE::base_pooling_type::setup_impl(inputs, outputs);
  auto inshape = inputs[0]->shape();
  const vector<int> int_inshape(inshape.cbegin(), inshape.cend());
  auto dtype = cudnn_data_type<T>::type();
  cudnn_pooling_ = CudnnPooling::create(
      int_inshape, this->kernel_, this->stride_, this->ignore_border_,
      this->pad_, this->channel_last_, this->mode(), dtype, device_);
}

template <typename BasePoolingType>
void BasePoolingCudaCudnn<BasePoolingType>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(cudnn_pooling_, error_code::value, "setup not called.");
  auto x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  cudnn_pooling_->forward(&alpha, x, &beta, y);
}

template <typename BasePoolingType>
void BasePoolingCudaCudnn<BasePoolingType>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  NBLA_CHECK(cudnn_pooling_, error_code::value, "setup not called.");
  auto dx = inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[0]);
  auto dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  // *y and *x are  not used in NNabla, but they are required with cudnn API
  auto y = outputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
  cudnn_pooling_->backward(&alpha, y, dy, x, &beta, dx);
}
} // namespace nbla
