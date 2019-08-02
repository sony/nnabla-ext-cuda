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

// deconvolution.cu

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/deconvolution.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void DeconvolutionCudaCudnn<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Deconvolution<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);

#if CUDNN_VERSION < 7000
  // Group stride of output
  x_offset_ = this->inner_size_i_ / this->group_;
  // Group stride of input
  y_offset_ = this->inner_size_o_ / this->group_;
  // Group stride of weight
  w_offset_ = this->channels_o_ * this->inner_size_k_ / this->group_;
  if (inputs.size() == 3) {
    b_offset_ = this->channels_i_ / this->group_;
  }
#endif

  // Create or query a resource.
  CudnnConvDesc desc{(int)this->kernel_.size(),
                     device_,
                     cudnn_data_type<T>::type(),
                     CUDNN_CROSS_CORRELATION,
                     this->outer_size_,
                     this->channels_i_,
                     this->channels_o_,
                     this->group_,
                     false, // TODO: Add channel_last option to deconv
                     this->spatial_shape_i_,
                     this->kernel_,
                     this->pad_,
                     this->stride_,
                     this->dilation_};
  auto &rsc = SingletonManager::get<CudnnHandleManager>()->conv_resource;
  auto it = rsc.find(desc);
  if (it != rsc.end()) {
    // Found a previously created one.
    // std::cout << "Found previously created one: " << desc << std::endl;
    rsc_ = it->second;
    return;
  }
  // Create a new resource.
  // This will search a best algorithm given config.
  rsc_ = make_shared<CudnnConvResource>(desc);
  rsc.insert({desc, rsc_}); // Register the created resource to global.
}

template <class T>
void DeconvolutionCudaCudnn<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tw *y = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  const Tw *w = inputs[1]->get_data_pointer<Tw>(this->ctx_);
  Tw *x = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  const Tw *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<Tw>(this->ctx_);
  }
  auto workspace_size = rsc_->workspace_size();
  unique_ptr<CudaCachedArray> workspace_arr;
  void *workspace{nullptr};
  if (workspace_size) {
    workspace_arr.reset(
        new CudaCachedArray(workspace_size, dtypes::BYTE, this->ctx_));
    workspace = workspace_arr->pointer<void>();
  }
#if CUDNN_VERSION >= 7000
  NBLA_CUDNN_CHECK(cudnnConvolutionBackwardData(
      cudnn_handle_, &alpha, rsc_->w_desc, w, rsc_->y_desc, y,
      rsc_->conv_dgrad_desc.desc, rsc_->bwd_data_algo, workspace,
      rsc_->bwd_data_workspace_size, &beta, rsc_->x_desc, x));
  if (inputs.size() == 3) {
    NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, rsc_->b_desc_deconv,
                                    b, &alpha, rsc_->x_desc, x));
  }
#else
  for (int g = 0; g < this->group_; ++g) {
    NBLA_CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_handle_, &alpha, rsc_->w_desc, w + w_offset_ * g, rsc_->y_desc,
        y + y_offset_ * g, rsc_->conv_dgrad_desc.desc, rsc_->bwd_data_algo,
        workspace, rsc_->bwd_data_workspace_size, &beta, rsc_->x_desc,
        x + x_offset_ * g));
    if (inputs.size() == 3) {
      // TODO: Bias addition should be outside of the loop. In that case,
      // b_desc and y_desc must be whole image descriptor.
      NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha,
                                      rsc_->b_desc_deconv, b + b_offset_ * g,
                                      &alpha, rsc_->x_desc, x + x_offset_ * g));
    }
  }
#endif
}

template <class T>
void DeconvolutionCudaCudnn<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {

  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tw *dx = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  const Tw *y;
  const Tw *w;
  Tw *dy, *dw, *db;
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<Tw>(this->ctx_);
    dy = inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[0]);
  }
  if (propagate_down[1]) {
    y = inputs[0]->get_data_pointer<Tw>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[1]);
  }
  if (propagate_down[2]) {
    db = inputs[2]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[2]);
  }
  auto alpha = get_cudnn_scalar_arg<T>(1);

  auto workspace_size = rsc_->workspace_size();
  unique_ptr<CudaCachedArray> workspace_arr;
  void *workspace{nullptr};
  if (workspace_size) {
    workspace_arr.reset(
        new CudaCachedArray(workspace_size, dtypes::BYTE, this->ctx_));
    workspace = workspace_arr->pointer<void>();
  }

#if CUDNN_VERSION >= 7000
  if (propagate_down[0]) {
    auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
    NBLA_CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_handle_, &alpha, rsc_->x_desc, dx, rsc_->w_desc, w,
        rsc_->conv_desc.desc, rsc_->fwd_algo, workspace,
        rsc_->fwd_workspace_size, &beta, rsc_->y_desc, dy));
  }
  if (propagate_down[1]) {
    auto beta = get_cudnn_scalar_arg<T>(accum[1] ? 1 : 0);
    NBLA_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        cudnn_handle_, &alpha, rsc_->x_desc, dx, rsc_->y_desc, y,
        rsc_->conv_wgrad_desc.desc, rsc_->bwd_filter_algo, workspace,
        rsc_->bwd_filter_workspace_size, &beta, rsc_->w_desc, dw));
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    auto beta = get_cudnn_scalar_arg<T>(accum[2] ? 1 : 0);
    NBLA_CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_handle_, &alpha,
                                                  rsc_->x_desc, dx, &beta,
                                                  rsc_->b_desc_deconv, db));
  }
#else
  for (int g = 0; g < this->group_; ++g) {
    if (propagate_down[0]) {
      auto beta = get_cudnn_scalar_arg<T>(accum[0] ? 1 : 0);
      // todo: will enable async streaming execution as well as convolution.
      NBLA_CUDNN_CHECK(cudnnConvolutionForward(
          cudnn_handle_, &alpha, rsc_->x_desc, dx + x_offset_ * g, rsc_->w_desc,
          w + w_offset_ * g, rsc_->conv_desc.desc, rsc_->fwd_algo, workspace,
          rsc_->fwd_workspace_size, &beta, rsc_->y_desc, dy + y_offset_ * g));
    }
    if (propagate_down[1]) {
      auto beta = get_cudnn_scalar_arg<T>(accum[1] ? 1 : 0);
      // todo: will enable async streaming execution as well as convolution.
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          cudnn_handle_, &alpha, rsc_->x_desc, dx + x_offset_ * g, rsc_->y_desc,
          y + y_offset_ * g, rsc_->conv_wgrad_desc.desc, rsc_->bwd_filter_algo,
          workspace, rsc_->bwd_filter_workspace_size, &beta, rsc_->w_desc,
          dw + w_offset_ * g));
    }
    if (inputs.size() == 3 && propagate_down[2]) {
      auto beta = get_cudnn_scalar_arg<T>(accum[2] ? 1 : 0);
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardBias(
          cudnn_handle_, &alpha, rsc_->x_desc, dx + x_offset_ * g, &beta,
          rsc_->b_desc_deconv, db + b_offset_ * g));
    }
  }
#endif
}
}
