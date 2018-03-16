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

#include <cstdlib>
#include <string>

namespace nbla {

void cudnn_set_tensor_nd_descriptor_force_dim(cudnnTensorDescriptor_t &desc,
                                              cudnnDataType_t dtype,
                                              vector<int> dims,
                                              vector<int> strides,
                                              int force_ndim) {
  if (dims.size() < force_ndim) {
    dims.resize(force_ndim, 1);
    strides.resize(force_ndim, 1);
  }
  NBLA_CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, dtype, dims.size(),
                                              dims.data(), strides.data()));
}

/* Wrapper function of cudnnSetConvolutionNdDescriptor with ensuring filter
 * dimension >= 1.
 */
inline void cudnn_set_convolution_nd_descriptor_force_2dim(
    cudnnConvolutionDescriptor_t &conv_desc, int ndim, vector<int> pad,
    vector<int> stride, vector<int> dilation, cudnnConvolutionMode_t mode,
    cudnnDataType_t dtype) {
  if (ndim == 1) {
    ndim = 2;
    pad.resize(2, 0);
    stride.resize(2, 1);
    dilation.resize(2, 1);
  }
  NBLA_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      conv_desc, ndim, pad.data(), stride.data(), dilation.data(), mode,
      dtype));
}

static size_t get_conv_outsize(int w, int k, int p, int s, int d) {
  const int dk = d * (k - 1) + 1;
  return (w + 2 * p - dk) / s + 1;
}

bool CudnnConvDesc::operator==(const CudnnConvDesc &x) const {
  if (ndim != x.ndim || device != x.device || dtype != x.dtype ||
      mode != x.mode || n != x.n || c != x.c || o != x.o || group != x.group)
    return false;
  for (int d = 0; d < ndim; d++) {
    if (sample[d] != x.sample[d] || kernel[d] != x.kernel[d] ||
        pad[d] != x.pad[d] || stride[d] != x.stride[d] ||
        dilation[d] != x.dilation[d])
      return false;
  }
  return true;
}

std::ostream &operator<<(std::ostream &os, const CudnnConvDesc &desc) {
  os << "[CudnnConvDesc]" << std::endl;
  os << "  ndim = " << desc.ndim << std::endl;
  os << "  device = " << desc.device << std::endl;
  os << "  dtype = " << (int)(desc.dtype) << std::endl;
  os << "  mode = " << (int)(desc.mode) << std::endl;
  os << "  n, c, o = " << desc.n << ", " << desc.c << ", " << desc.o
     << std::endl;
  os << "  group = " << desc.group << std::endl;
  for (int d = 0; d < desc.ndim; d++) {
    os << "  d, k, p, s, d = " << desc.sample[d] << " " << desc.kernel[d] << " "
       << desc.pad[d] << " " << desc.stride[d] << " " << desc.dilation[d]
       << std::endl;
  }
  return os;
}

inline vector<int> get_conv_dims(int b, int c, int group,
                                 const vector<int> &sample) {
  vector<int> ret(sample.size() + 2);
  ret[0] = b;
  ret[1] = c / group;
  for (int d = 0; d < sample.size(); d++) {
    ret[d + 2] = sample[d];
  }
  return ret;
}

inline vector<int> get_conv_strides(int c, const vector<int> &sample) {
  vector<int> ret(sample.size() + 2);
  int stride = 1;
  for (int d = sample.size() - 1; d >= 0; d--) {
    ret[d + 2] = stride;
    stride *= sample[d];
  }
  ret[1] = stride;
  ret[0] = stride * c;
  return ret;
}

CudnnConvResource::CudnnConvResource(const CudnnConvDesc &desc) {
  // std::cout << "Creating resource for: " << desc << std::endl;
  device = desc.device;
  // Allocate memory for descriptors
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
  NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc_deconv));
  NBLA_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
  NBLA_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  // Set input desc
  auto dims = get_conv_dims(desc.n, desc.c, desc.group, desc.sample);
  auto strides = get_conv_strides(desc.c, desc.sample);
  cudnn_set_tensor_nd_descriptor_force_dim(x_desc, desc.dtype, dims, strides,
                                           4);

  // Set output desc
  vector<int> osample(desc.ndim);
  for (int d = 0; d < desc.ndim; d++) {
    osample[d] = get_conv_outsize(desc.sample[d], desc.kernel[d], desc.pad[d],
                                  desc.stride[d], desc.dilation[d]);
  }
  auto odims = get_conv_dims(desc.n, desc.o, desc.group, osample);
  auto ostrides = get_conv_strides(desc.o, osample);
  cudnn_set_tensor_nd_descriptor_force_dim(y_desc, desc.dtype, odims, ostrides,
                                           4);

  // Set kernel (filter) desc
  vector<int> fdims(desc.ndim + 2);
  fdims[0] = desc.o / desc.group;
  fdims[1] = desc.c / desc.group;
  for (int d = 0; d < desc.ndim; d++) {
    fdims[d + 2] = desc.kernel[d];
  }
  if (desc.ndim == 1) {
    fdims.resize(4, 1);
  }
  NBLA_CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      w_desc, desc.dtype, CUDNN_TENSOR_NCHW, fdims.size(), fdims.data()));

  // Set bias desc
  vector<int> bdims(desc.ndim + 2, 1);
  bdims[1] = desc.o / desc.group;
  vector<int> bstrides(desc.ndim + 2, 1);
  bstrides[0] = bdims[1];
  cudnn_set_tensor_nd_descriptor_force_dim(b_desc, desc.dtype, bdims, bstrides,
                                           4);

  // Set bias desc for deconvolution
  bdims[1] = desc.c / desc.group;
  bstrides[0] = bdims[1];
  cudnn_set_tensor_nd_descriptor_force_dim(b_desc_deconv, desc.dtype, bdims,
                                           bstrides, 4);

  // Set Conv desc
  // TODO: Support compute type config
  cudnn_set_convolution_nd_descriptor_force_2dim(
      conv_desc, desc.ndim, desc.pad, desc.stride, desc.dilation, desc.mode,
      cudnn_data_type<float>::type());

  // Find best algorithm
  find_best_algorithms();
}

CudnnConvResource::~CudnnConvResource() {

  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(b_desc_deconv));
  NBLA_CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
  NBLA_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

void CudnnConvResource::find_best_algorithms_no_limit() {
#if CUDNN_VERSION >= 3000
  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(device);
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
#else
  // TODO: Logger?
  std::cout << "CUDNN version less than 3000 doesn't have "
               "cudnnFindConvolution*Algorithm. Fallen back to no-workspace "
               "mode. If you want a more faster convolution, you should set "
               "NNABLA_CUDNN_WORKSPACE_LIMIT greater than 0 as an environment "
               "variable."
            << std::endl;
  find_best_algorithms_no_workspace();
#endif
}

void CudnnConvResource::find_best_algorithms_limit(int limit) {
  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(device);
  // Get forward algorithm
  NBLA_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle, x_desc, w_desc, conv_desc, y_desc,
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, limit, &fwd_algo));
  // Get data backward workspace
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, w_desc, y_desc, conv_desc, x_desc,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, limit,
      &bwd_data_algo));
  // Get filter backward workspace
  NBLA_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle, x_desc, y_desc, conv_desc, w_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, limit,
      &bwd_filter_algo));
  // Set workspace
  fwd_workspace_size = limit;
  bwd_data_workspace_size = limit;
  bwd_filter_workspace_size = limit;
}

void CudnnConvResource::find_best_algorithms_no_workspace() {
  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(device);
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
  fwd_workspace_size = 0;
  bwd_data_workspace_size = 0;
  bwd_filter_workspace_size = 0;
}

void CudnnConvResource::find_best_algorithms() {
  int workspace_limit = SingletonManager::get<CudnnHandleManager>()
                            ->get_workspace_limit_in_bytes();
  if (workspace_limit < 0) {
    find_best_algorithms_no_limit();
  } else if (workspace_limit == 0) {
    find_best_algorithms_no_workspace();
  } else {
    find_best_algorithms_limit(workspace_limit);
  }
}

size_t CudnnConvResource::workspace_size() const {
  return std::max(fwd_workspace_size,
                  std::max(bwd_filter_workspace_size, bwd_data_workspace_size));
}

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

int CudnnHandleManager::get_workspace_limit_in_bytes() {
  static bool called = false;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (!called) {
    // trying to get a default value from env var.
    const char *e = std::getenv("NNABLA_CUDNN_WORKSPACE_LIMIT");
    if (!e) {
      workspace_limit_ = -1; // default is no limit.
    } else {
      try {
        workspace_limit_ = std::stoi(e);
      } catch (std::exception &exc) {
        NBLA_ERROR(
            error_code::value,
            "Invalid value: NNBLA_CUDNN_WORKSPACE_LIMIT=%s. Integer required.",
            e);
      }
    }
    called = true;
  }
  return workspace_limit_;
}

void CudnnHandleManager::set_workspace_limit_in_bytes(int bytes) {
  workspace_limit_ = bytes;
}

NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, CudnnHandleManager);
}
