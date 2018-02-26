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

// convolution_cudnn.cu

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/convolution.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void ConvolutionCudaCudnn<T>::setup_impl(const Variables &inputs,
                                         const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Convolution<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  if (this->spatial_dims_ == 2) {
    setup_impl_2d(inputs, outputs);
  } else {
    setup_impl_nd(inputs, outputs);
  }
}

template <typename T>
void ConvolutionCudaCudnn<T>::setup_impl_2d(const Variables &inputs,
                                            const Variables &outputs) {
  // Set input tensor descriptor
  /*
  const int x_n = this->outer_size_;
  const int x_c = this->channels_i_ / this->group_;
  const int x_h = this->spatial_shape_i_[0];
  const int x_w = this->spatial_shape_i_[1];
  */
  x_offset_ = this->inner_size_i_ / this->group_;
  // Set output tensor descriptor
  /*
  const int y_n = this->outer_size_;
  const int y_c = this->channels_o_ / this->group_;
  const int y_h = this->spatial_shape_o_[0];
  const int y_w = this->spatial_shape_o_[1];
  */
  y_offset_ = this->inner_size_o_ / this->group_;
  // Set filter kernel descriptor
  /*
  const int w_k = this->channels_o_ / this->group_;
  const int w_c = this->channels_g_;
  const int w_h = this->kernel_[0];
  const int w_w = this->kernel_[1];
  */
  w_offset_ = this->channels_o_ * this->inner_size_k_ / this->group_;
  // Set bias tensor descriptor
  if (inputs.size() == 3) {
    b_offset_ = this->channels_o_ / this->group_;
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
void ConvolutionCudaCudnn<T>::setup_impl_nd(const Variables &inputs,
                                            const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented,
             "2D convolution is only supported so far.");
}

template <class T>
void ConvolutionCudaCudnn<T>::forward_impl(const Variables &inputs,
                                           const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T alpha = 1;
  T beta = 0;
  const T *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<T>(this->ctx_);
  }
  void *workspace = SingletonManager::get<Cuda>()->get_workspace(
      rsc2d_->workspace_size(), this->device_);
  for (int g = 0; g < this->group_; ++g) {
    NBLA_CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_handle_, &alpha, rsc2d_->x_desc, x + x_offset_ * g,
        rsc2d_->w_desc, w + w_offset_ * g, rsc2d_->conv_desc, rsc2d_->fwd_algo,
        workspace, rsc2d_->fwd_workspace_size, &beta, rsc2d_->y_desc,
        y + y_offset_ * g));
    if (inputs.size() == 3) {
      // TODO: Bias addition should be outside of the loop. In that case,
      // b_desc and y_desc must be whole image descriptor.
      NBLA_CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha, rsc2d_->b_desc,
                                      b + b_offset_ * g, &alpha, rsc2d_->y_desc,
                                      y + y_offset_ * g));
    }
  }
}

template <class T>
void ConvolutionCudaCudnn<T>::backward_impl(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) {

  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x;
  const T *w;
  T *dx, *dw, *db;
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<T>(this->ctx_);
    dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[1]) {
    x = inputs[0]->get_data_pointer<T>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  if (propagate_down[2]) {
    db = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_);
  }
  T alpha = 1;
  void *workspace = SingletonManager::get<Cuda>()->get_workspace(
      rsc2d_->workspace_size(), this->device_);
  for (int g = 0; g < this->group_; ++g) {
    if (propagate_down[0]) {
      T beta = accum[0] ? 1 : 0;
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardData(
          cudnn_handle_, &alpha, rsc2d_->w_desc, w + w_offset_ * g,
          rsc2d_->y_desc, dy + y_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->bwd_data_algo, workspace, rsc2d_->bwd_data_workspace_size,
          &beta, rsc2d_->x_desc, dx + x_offset_ * g));
    }
    if (propagate_down[1]) {
      T beta = accum[1] ? 1 : 0;
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          cudnn_handle_, &alpha, rsc2d_->x_desc, x + x_offset_ * g,
          rsc2d_->y_desc, dy + y_offset_ * g, rsc2d_->conv_desc,
          rsc2d_->bwd_filter_algo, workspace, rsc2d_->bwd_filter_workspace_size,
          &beta, rsc2d_->w_desc, dw + w_offset_ * g));
    }
    if (inputs.size() == 3 && propagate_down[2]) {
      T beta = accum[2] ? 1 : 0;
      NBLA_CUDNN_CHECK(cudnnConvolutionBackwardBias(
          cudnn_handle_, &alpha, rsc2d_->y_desc, dy + y_offset_ * g, &beta,
          rsc2d_->b_desc, db + b_offset_ * g));
    }
  }
}

// Manually selecting algorithms is not supported for now.
/*
// Basically this functions is not invoked,
// because it is choosen by cudnnGetConvolutionForwardAlgorithm()
template <class T>
void ConvolutionCudaCudnn<T>::set_cudnn_convolution_forward_algorithm(
    std::string algorithm) {
  if (algorithm == "ALGO_IMPLICIT_GEMM") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  } else if (algorithm == "ALGO_IMPLICIT_PRECOMP_GEMM") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  } else if (algorithm == "ALGO_GEMM") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  } else if (algorithm == "ALGO_DIRECT") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  } else if (algorithm == "ALGO_FFT") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  } else if (algorithm == "ALGO_FFT_TILING") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  }
#if CUDNN_VERSION >= 5000
  else if (algorithm == "ALGO_WINOGRAD") {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  }
#endif
  else {
    NBLA_ERROR(error_code::target_specific,
               "Specified unsupported forward algorithm");
  }
}

// Basically this functions is not invoked,
// because it is choosen by cudnnGetConvolutionBackwardFilterAlgorithm()
template <class T>
void ConvolutionCudaCudnn<T>::set_cudnn_convolution_backward_filter_algorithm(
    std::string algorithm) {
  if (algorithm == "ALGO_0") {
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  } else if (algorithm == "ALGO_1") {
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  } else if (algorithm == "ALGO_FFT") {
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
  } else if (algorithm == "ALGO_3") {
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
  }
#if CUDNN_VERSION >= 5100
  else if (algorithm == "ALGO_WINOGRAD_NONFUSED") {
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
  }
#endif
  else {
    NBLA_ERROR(error_code::target_specific,
               "Specified unsupported backward filter algorithm");
  }
}

// Basically this functions is not invoked,
// because it is choosen by cudnnGetConvolutionBackwardDataAlgorithm()
template <class T>
void ConvolutionCudaCudnn<T>::set_cudnn_convolution_backward_data_algorithm(
    std::string algorithm) {
  if (algorithm == "ALGO_0") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  } else if (algorithm == "ALGO_1") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  } else if (algorithm == "ALGO_FFT") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  } else if (algorithm == "ALGO_FFT_TILING") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
  }
#if CUDNN_VERSION >= 5000
  else if (algorithm == "ALGO_WINOGRAD") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
  }
#if CUDNN_VERSION >= 5100
  else if (algorithm == "ALGO_WINOGRAD_NONFUSED") {
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
  }
#endif
#endif
  else {
    NBLA_ERROR(error_code::target_specific,
               "Specified unsupported backward data algorithm");
  }
}

template <class T>
void ConvolutionCudaCudnn<T>::set_cudnn_convolution_mode(std::string mode) {
  if (mode == "CONVOLUTION") {
    mode_ = CUDNN_CONVOLUTION;
  } else if (mode == "CROSS_CORRELATION") {
    mode_ = CUDNN_CROSS_CORRELATION;
  } else {
    NBLA_ERROR(error_code::target_specific, "Specified unsupported algorithm");
  }
}
*/

template class ConvolutionCudaCudnn<float>;
}
