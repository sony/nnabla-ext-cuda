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
  if (this->spatial_dims_ == 2) {
    setup_impl_2d(inputs, outputs);
  } else {
    setup_impl_nd(inputs, outputs);
  }
}

template <typename T>
void DeconvolutionCudaCudnn<T>::setup_impl_2d(const Variables &inputs,
                                              const Variables &outputs) {
  // Group stride of output
  x_offset_ = this->inner_size_i_ / this->group_;
  // Group stride of input
  y_offset_ = this->inner_size_o_ / this->group_;
  // Group stride of weight
  w_offset_ = this->channels_o_ * this->inner_size_k_ / this->group_;
  if (inputs.size() == 3) {
    b_offset_ = this->channels_i_ / this->group_;
  }

  // Create or query a resource.
  CudnnConv2dDesc desc{device_,
                       cudnn_data_type<T>::type(),
                       CUDNN_CROSS_CORRELATION,
                       this->outer_size_,
                       this->channels_i_,
                       this->spatial_shape_i_[0],
                       this->spatial_shape_i_[1],
                       this->channels_o_,
                       this->kernel_[0],
                       this->kernel_[1],
                       this->pad_[0],
                       this->pad_[1],
                       this->stride_[0],
                       this->stride_[1],
                       this->group_,
                       this->dilation_[0],
                       this->dilation_[1]};
  auto &rsc = SingletonManager::get<CudnnHandleManager>()->conv2d_resource;
  auto it = rsc.find(desc);
  if (it != rsc.end()) {
    // Found a previously created one.
    // std::cout << "Found previously created one: " << desc << std::endl;
    rsc2d_ = it->second;
    return;
  }
  // Create a new resource.
  // This will search a best algorithm given config.
  rsc2d_ = make_shared<CudnnConv2dResource>(desc);
  rsc.insert({desc, rsc2d_}); // Register the created resource to global.
}

template <class T>
void DeconvolutionCudaCudnn<T>::setup_impl_nd(const Variables &inputs,
                                              const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented,
             "2D deconvolution is only supported so far.");
}

template <class T>
void DeconvolutionCudaCudnn<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *y = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *x = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = 0;
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }
#ifdef NBLA_CUDNN_USE_WORKSPACE
  void *workspace = SingletonManager::get<Cuda>()->get_workspace(
      rsc2d_->workspace_size(), this->device_);
#endif
  for (int g = 0; g < this->group_; ++g) {
#ifdef NBLA_CUDNN_USE_WORKSPACE
    NBLA_CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_handle_, &alpha, rsc2d_->w_desc, w + w_offset_ * g,
        rsc2d_->y_desc, y + y_offset_ * g, rsc2d_->conv_desc,
        rsc2d_->bwd_data_algo, workspace, rsc2d_->bwd_data_workspace_size,
        &beta, rsc2d_->x_desc, x + x_offset_ * g));
#else
    NBLA_CUDNN_CHECK(cudnnConvolutionBackwardData(
        cudnn_handle_, &alpha, rsc2d_->w_desc, w + w_offset_ * g,
        rsc2d_->y_desc, y + y_offset_ * g, rsc2d_->conv_desc,
        rsc2d_->bwd_data_algo, NULL, 0, &beta, rsc2d_->x_desc,
        x + x_offset_ * g));
#endif
    if (inputs.size() == 3) {
      // TODO: Bias addition should be outside of the loop. In that case,
      // b_desc and y_desc must be whole image descriptor.
      NBLA_CUDNN_CHECK(cudnnAddTensor(
          cudnn_handle_, &alpha, rsc2d_->b_desc_deconv, b + b_offset_ * g,
          &alpha, rsc2d_->x_desc, x + x_offset_ * g));
    }
  }
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
  const T *dx = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *y;
  const T *w;
  T *dy, *dw, *db;
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dy = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[1]) {
    y = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[2]) {
    db = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  T alpha = 1;
#ifdef NBLA_CUDNN_USE_WORKSPACE
  void *workspace = SingletonManager::get<Cuda>()->get_workspace(
      rsc2d_->workspace_size(), this->device_);
#endif
  for (int g = 0; g < this->group_; ++g) {
    if (propagate_down[0]) {
      T beta = accum[0] ? 1 : 0;
#ifdef NBLA_CUDNN_USE_WORKSPACE
      NBLA_CUDNN_CHECK(cudnnConvolutionForward(
          cudnn_handle_, &alpha, rsc2d_->x_desc, dx + x_offset_ * g,
          rsc2d_->w_desc, w + w_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->fwd_algo, workspace, rsc2d_->fwd_workspace_size, &beta,
          rsc2d_->y_desc, dy + y_offset_ * g));
#else
      NBLA_CUDNN_CHECK(cudnnConvolutionForward(
          cudnn_handle_, &alpha, rsc2d_->x_desc, dx + x_offset_ * g,
          rsc2d_->w_desc, w + w_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->fwd_algo, NULL, 0, &beta, rsc2d_->y_desc,
          dy + y_offset_ * g));
#endif
    }
    if (propagate_down[1]) {
      T beta = accum[1] ? 1 : 0;
#ifdef NBLA_CUDNN_USE_WORKSPACE
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          cudnn_handle_, &alpha, rsc2d_->x_desc, dx + x_offset_ * g,
          rsc2d_->y_desc, y + y_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->bwd_filter_algo, workspace, rsc2d_->bwd_filter_workspace_size,
          &beta, rsc2d_->w_desc, dw + w_offset_ * g));
#else
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          cudnn_handle_, &alpha, rsc2d_->x_desc, dx + x_offset_ * g,
          rsc2d_->y_desc, y + y_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->bwd_filter_algo, NULL, 0, &beta, rsc2d_->w_desc,
          dw + w_offset_ * g));
#endif
    }
    if (inputs.size() == 3 && propagate_down[2]) {
      T beta = accum[2] ? 1 : 0;
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardBias(
          cudnn_handle_, &alpha, rsc2d_->x_desc, dx + x_offset_ * g, &beta,
          rsc2d_->b_desc_deconv, db + b_offset_ * g));
    }
  }
}

template class DeconvolutionCudaCudnn<float>;
}
