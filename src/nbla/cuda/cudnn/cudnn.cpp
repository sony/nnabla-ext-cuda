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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/singleton_manager-internal.hpp>

// #include <iostream>  // TODO: remove

namespace nbla {

static size_t get_conv_outsize(int w, int k, int p, int s) {
  return (w + 2 * p - k) / s + 1;
}

bool CudnnConv2dDesc::operator==(const CudnnConv2dDesc &x) const {
  return device == x.device && dtype == x.dtype && mode == x.mode && n == x.n &&
         c == x.c && h == x.h && w == x.w && o == x.o && kh == x.kh &&
         kw == x.kw && padh == x.padh && padw == x.padw &&
         strideh == x.strideh && stridew == x.stridew && group == x.group;
}

std::ostream &operator<<(std::ostream &os, const CudnnConv2dDesc &desc) {
  os << "[CudnnConv2dDesc]" << std::endl;
  os << "  device = " << desc.device << std::endl;
  os << "  dtype = " << (int)(desc.dtype) << std::endl;
  os << "  mode = " << (int)(desc.mode) << std::endl;
  os << "  n, c, h, w = " << desc.n << ", " << desc.c << ", " << desc.h << ", "
     << desc.w << std::endl;
  os << "  o, c, kh, kw = " << desc.o << ", " << desc.c << ", " << desc.kh
     << ", " << desc.kw << std::endl;
  os << "  padh, padw = " << desc.padh << ", " << desc.padw << std::endl;
  os << "  strideh, stridew = " << desc.strideh << ", " << desc.stridew
     << std::endl;
  os << "  group = " << desc.group << std::endl;
  return os;
}

CudnnConv2dResource::CudnnConv2dResource(const CudnnConv2dDesc &desc) {
  // std::cout << "Creating resource for: " << desc << std::endl;
  device = desc.device;
  // Allocate memory for descriptors
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
  NBLA_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
  NBLA_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  // Set input desc
  const int s_w = 1;
  const int s_h = desc.w;
  const int s_c = desc.h * s_h;
  const int s_n = desc.c * s_c;
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(x_desc, desc.dtype, desc.n,
                                                desc.c / desc.group, desc.h,
                                                desc.w, s_n, s_c, s_h, s_w));
  // Set output desc
  int ho = get_conv_outsize(desc.h, desc.kh, desc.padh, desc.strideh);
  int wo = get_conv_outsize(desc.w, desc.kw, desc.padw, desc.stridew);
  const int so_w = 1;
  const int so_h = wo;
  const int so_c = ho * so_h;
  const int so_n = so_c * desc.o;
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(y_desc, desc.dtype, desc.n,
                                                desc.o / desc.group, ho, wo,
                                                so_n, so_c, so_h, so_w));
// Set kernel desc
#if CUDNN_VERSION >= 5000
  NBLA_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      w_desc, desc.dtype, CUDNN_TENSOR_NCHW, desc.o / desc.group,
      desc.c / desc.group, desc.kh, desc.kw));
#else
  NBLA_CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(
      w_desc, desc.dtype, CUDNN_TENSOR_NCHW, desc.o / desc.group,
      desc.c / desc.group, desc.kh, desc.kw));
#endif
  // Set bias desc
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      b_desc, CUDNN_TENSOR_NCHW, desc.dtype, 1, desc.o / desc.group, 1, 1));
#if CUDNN_VERSION >= 6000
  // Set Conv desc
  // TODO: Support data type config ang dilated convolution.
  NBLA_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, desc.padh, desc.padw, desc.strideh, desc.stridew, 1, 1,
      desc.mode, cudnn_data_type<float>::type()));
#else
  // Set Conv desc
  NBLA_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, desc.padh, desc.padw, desc.strideh, desc.stridew, 1, 1,
      desc.mode));
#endif
  find_best_algorithms();
}

CudnnConv2dResource::~CudnnConv2dResource() {

  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

void CudnnConv2dResource::find_best_algorithms() {
  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(device);
#ifdef NBLA_CUDNN_USE_WORKSPACE
  // Get forward algorithm
  int fwd_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t fwd_data_perf_results[1];
  NBLA_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
      cudnn_handle, x_desc, w_desc, conv_desc, y_desc, 1, &fwd_algo_count,
      fwd_data_perf_results));
  fwd_algo = fwd_data_perf_results[0].algo;
  // Get forward workspace size
  NBLA_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, x_desc, w_desc, conv_desc, y_desc, fwd_algo,
      &fwd_workspace_size));
  // Choose data backward algorithm
  int bwd_data_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf_results[1];
  NBLA_CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
      cudnn_handle, w_desc, y_desc, conv_desc, x_desc, 1, &bwd_data_algo_count,
      bwd_data_perf_results));
  bwd_data_algo = bwd_data_perf_results[0].algo;
  // Get data backward workspace size
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, w_desc, y_desc, conv_desc, x_desc, bwd_data_algo,
      &bwd_data_workspace_size));
  // Choose filter backward algorithm
  int bwd_filter_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf_results[1];
  NBLA_CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
      cudnn_handle, x_desc, y_desc, conv_desc, w_desc, 1,
      &bwd_filter_algo_count, bwd_filter_perf_results));
  bwd_filter_algo = bwd_filter_perf_results[0].algo;
  // Get filter backward workspace size
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle, x_desc, y_desc, conv_desc, w_desc, bwd_filter_algo,
      &bwd_filter_workspace_size));
#else // NOT_USE_WORKSPACE
  // Get forward algorithm
  NBLA_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
      CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &fwd_algo));
  // Get data backward workspace
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, w_desc, y_desc, conv_desc, x_desc,
      CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &bwd_data_algo));
  // Get filter backward workspace
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, x_desc, y_desc, conv_desc, w_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &bwd_filter_algo));
#endif
}

#ifdef NBLA_CUDNN_USE_WORKSPACE
size_t CudnnConv2dResource::workspace_size() const {
  return std::max(fwd_workspace_size,
                  std::max(bwd_filter_workspace_size, bwd_data_workspace_size));
}
#endif

//////////////////////////////
// cuDNN Handle implementation
//////////////////////////////
CudnnHandleManager::CudnnHandleManager() {}

CudnnHandleManager::~CudnnHandleManager() {
  for (auto handle : this->handles_) {
    NBLA_CUDNN_CHECK(cudnnDestroy(handle.second));
  }
}

cudnnHandle_t CudnnHandleManager::handle(int device) {
  if (device < 0) {
    NBLA_CUDA_CHECK(cudaGetDevice(&device));
  }
  if (this->handles_.count(device) == 0) {
    cudnnHandle_t handle;
    NBLA_CUDNN_CHECK(cudnnCreate(&handle));
    this->handles_[device] = handle;
  }
  return this->handles_[device];
}
NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, CudnnHandleManager);
}
