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
    vector<int> stride, vector<int> dilation, int group,
    cudnnConvolutionMode_t mode, cudnnDataType_t dtype) {
  if (ndim == 1) {
    ndim = 2;
    pad.resize(2, 0);
    stride.resize(2, 1);
    dilation.resize(2, 1);
  }
#if CUDNN_VERSION >= 7000
  NBLA_CUDNN_CHECK(
      cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
  NBLA_CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
      conv_desc, ndim, pad.data(), stride.data(), dilation.data(), mode,
      dtype));
#if CUDNN_VERSION >= 7000
  NBLA_CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, group));
#endif
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
#if CUDNN_VERSION >= 7000
  ret[1] = c;
#else
  ret[1] = c / group;
#endif
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
#if CUDNN_VERSION >= 7000
  fdims[0] = desc.o;
#else
  fdims[0] = desc.o / desc.group;
#endif
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
#if CUDNN_VERSION >= 7000
  bdims[1] = desc.o;
#else
  bdims[1] = desc.o / desc.group;
#endif
  vector<int> bstrides(desc.ndim + 2, 1);
  bstrides[0] = bdims[1];
  cudnn_set_tensor_nd_descriptor_force_dim(b_desc, desc.dtype, bdims, bstrides,
                                           4);

// Set bias desc for deconvolution
#if CUDNN_VERSION >= 7000
  bdims[1] = desc.c;
#else
  bdims[1] = desc.c / desc.group;
#endif
  bstrides[0] = bdims[1];
  cudnn_set_tensor_nd_descriptor_force_dim(b_desc_deconv, desc.dtype, bdims,
                                           bstrides, 4);

  // Set Conv desc
  // TODO: Support compute type config (but... TRUE_HALF_CONFIG is not supported
  // in most of algorithms.)
  // Use FLOAT computation when HALF data type.
  cudnnDataType_t compute_type =
      desc.dtype == CUDNN_DATA_HALF ? CUDNN_DATA_FLOAT : desc.dtype;
  cudnn_set_convolution_nd_descriptor_force_2dim(
      conv_desc, desc.ndim, desc.pad, desc.stride, desc.dilation, desc.group,
      desc.mode, compute_type);

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

inline bool check_workspace_limit(int workspace_limit, size_t used_memory) {
  return (workspace_limit < 0) || (workspace_limit >= used_memory);
}

inline bool check_determinism(bool required, cudnnDeterminism_t determinism) {
  return (!required) || (determinism == CUDNN_DETERMINISTIC);
}

void CudnnConvResource::find_forward_algorithm(int workspace_limit,
                                               bool deterministic) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_max_count = cudnnGetConvolutionForwardAlgorithmMaxCount;
  auto find_algorithm = cudnnFindConvolutionForwardAlgorithm;
  int max_results, num_results;

  NBLA_CUDNN_CHECK(get_max_count(cudnn_handle, &max_results));

  std::unique_ptr<cudnnConvolutionFwdAlgoPerf_t[]> perf_results{
      new cudnnConvolutionFwdAlgoPerf_t[max_results]};

  NBLA_CUDNN_CHECK(find_algorithm(cudnn_handle, this->x_desc, this->w_desc,
                                  this->conv_desc, this->y_desc, max_results,
                                  &num_results, perf_results.get()));
#if 0
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (perf_result.status == 0) {
      printf("fwd_algo %d took %f ms using %10lu bytes and is %s\n",
             perf_result.algo, perf_result.time, perf_result.memory,
             (perf_result.determinism ? "deterministic" : "not deterministic"));
    }
  }
#endif
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (CUDNN_STATUS_SUCCESS == perf_result.status) {
      if (check_workspace_limit(workspace_limit, perf_result.memory)) {
        if (check_determinism(deterministic, perf_result.determinism)) {
          this->fwd_algo = perf_result.algo;
          this->fwd_workspace_size = perf_result.memory;
          return;
        }
      }
    }
  }
  NBLA_ERROR(error_code::target_specific,
             "Could not find any CUDNN Convolution Forward Algorithm for "
             "the combination of NNBLA_CUDNN_WORKSPACE_LIMIT=%d and "
             "NNABLA_CUDNN_DETERMINISTIC=%d",
             workspace_limit, deterministic);
}

void CudnnConvResource::find_backward_data_algorithm(int workspace_limit,
                                                     bool deterministic) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_max_count = cudnnGetConvolutionBackwardDataAlgorithmMaxCount;
  auto find_algorithm = cudnnFindConvolutionBackwardDataAlgorithm;
  int max_results, num_results;

  NBLA_CUDNN_CHECK(get_max_count(cudnn_handle, &max_results));

  std::unique_ptr<cudnnConvolutionBwdDataAlgoPerf_t[]> perf_results{
      new cudnnConvolutionBwdDataAlgoPerf_t[max_results]};

  NBLA_CUDNN_CHECK(find_algorithm(cudnn_handle, this->w_desc, this->y_desc,
                                  this->conv_desc, this->x_desc, max_results,
                                  &num_results, perf_results.get()));
#if 0
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (perf_result.status == 0) {
      printf("bwd_data_algo %d took %f ms using %10lu bytes and is %s\n",
             perf_result.algo, perf_result.time, perf_result.memory,
             (perf_result.determinism ? "deterministic" : "not deterministic"));
    }
  }
#endif
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (CUDNN_STATUS_SUCCESS == perf_result.status) {
      if (check_workspace_limit(workspace_limit, perf_result.memory)) {
        if (check_determinism(deterministic, perf_result.determinism)) {
          this->bwd_data_algo = perf_result.algo;
          this->bwd_data_workspace_size = perf_result.memory;
          return;
        }
      }
    }
  }
  NBLA_ERROR(error_code::target_specific,
             "Could not find any CUDNN Convolution Backward Data Algorithm "
             "for the combination of NNBLA_CUDNN_WORKSPACE_LIMIT=%d and "
             "NNABLA_CUDNN_DETERMINISTIC=%d",
             workspace_limit, deterministic);
}

void CudnnConvResource::find_backward_filter_algorithm(int workspace_limit,
                                                       bool deterministic) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_max_count = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount;
  auto find_algorithm = cudnnFindConvolutionBackwardFilterAlgorithm;
  int max_results, num_results;

  NBLA_CUDNN_CHECK(get_max_count(cudnn_handle, &max_results));

  std::unique_ptr<cudnnConvolutionBwdFilterAlgoPerf_t[]> perf_results{
      new cudnnConvolutionBwdFilterAlgoPerf_t[max_results]};

  NBLA_CUDNN_CHECK(find_algorithm(cudnn_handle, this->x_desc, this->y_desc,
                                  this->conv_desc, this->w_desc, max_results,
                                  &num_results, perf_results.get()));
#if 0
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (perf_result.status == 0) {
      printf("bwd_filter_algo %d took %f ms using %10lu bytes and is %s\n",
             perf_result.algo, perf_result.time, perf_result.memory,
             (perf_result.determinism ? "deterministic" : "not deterministic"));
    }
  }
#endif
  for (int i = 0; i < num_results; i++) {
    auto &perf_result = perf_results[i];
    if (CUDNN_STATUS_SUCCESS == perf_result.status) {
      if (check_workspace_limit(workspace_limit, perf_result.memory)) {
        if (check_determinism(deterministic, perf_result.determinism)) {
          this->bwd_filter_algo = perf_result.algo;
          this->bwd_filter_workspace_size = perf_result.memory;
          return;
        }
      }
    }
  }
  NBLA_ERROR(error_code::target_specific,
             "Could not find any CUDNN Convolution Backward Filter Algorithm "
             "for the combination of NNBLA_CUDNN_WORKSPACE_LIMIT=%d and "
             "NNABLA_CUDNN_DETERMINISTIC=%d",
             workspace_limit, deterministic);
}

void CudnnConvResource::get_forward_algorithm(int workspace_limit) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_algorithm = cudnnGetConvolutionForwardAlgorithm;
  auto get_workspace = cudnnGetConvolutionForwardWorkspaceSize;

  auto preference = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
  if (workspace_limit < 0)
    preference = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
  else if (workspace_limit > 0)
    preference = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;

  NBLA_CUDNN_CHECK(get_algorithm(cudnn_handle, this->x_desc, this->w_desc,
                                 this->conv_desc, this->y_desc, preference,
                                 workspace_limit, &this->fwd_algo));
  if (workspace_limit != 0) {
    NBLA_CUDNN_CHECK(get_workspace(cudnn_handle, this->x_desc, this->w_desc,
                                   this->conv_desc, this->y_desc,
                                   this->fwd_algo, &this->fwd_workspace_size));
  } else {
    this->fwd_workspace_size = 0;
  }
}

void CudnnConvResource::get_backward_data_algorithm(int workspace_limit) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_algorithm = cudnnGetConvolutionBackwardDataAlgorithm;
  auto get_workspace = cudnnGetConvolutionBackwardDataWorkspaceSize;

  auto preference = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
  if (workspace_limit < 0)
    preference = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
  else if (workspace_limit > 0)
    preference = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;

  NBLA_CUDNN_CHECK(get_algorithm(cudnn_handle, this->w_desc, this->y_desc,
                                 this->conv_desc, this->x_desc, preference,
                                 workspace_limit, &this->bwd_data_algo));
  if (workspace_limit != 0) {
    NBLA_CUDNN_CHECK(get_workspace(
        cudnn_handle, this->w_desc, this->y_desc, this->conv_desc, this->x_desc,
        this->bwd_data_algo, &this->bwd_data_workspace_size));
  } else {
    this->bwd_data_workspace_size = 0;
  }
}

void CudnnConvResource::get_backward_filter_algorithm(int workspace_limit) {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(device);
  auto get_algorithm = cudnnGetConvolutionBackwardFilterAlgorithm;
  auto get_workspace = cudnnGetConvolutionBackwardFilterWorkspaceSize;

  auto preference = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
  if (workspace_limit < 0)
    preference = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
  else if (workspace_limit > 0)
    preference = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;

  NBLA_CUDNN_CHECK(get_algorithm(cudnn_handle, this->x_desc, this->y_desc,
                                 this->conv_desc, this->w_desc, preference,
                                 workspace_limit, &this->bwd_filter_algo));
  if (workspace_limit != 0) {
    NBLA_CUDNN_CHECK(get_workspace(
        cudnn_handle, this->x_desc, this->y_desc, this->conv_desc, this->w_desc,
        this->bwd_filter_algo, &this->bwd_filter_workspace_size));
  } else {
    this->bwd_filter_workspace_size = 0;
  }
}

void CudnnConvResource::find_best_algorithms() {
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto workspace_limit = cudnn_handle_manager->get_workspace_limit_in_bytes();
  bool deterministic = cudnn_handle_manager->get_deterministic_option();

#if CUDNN_VERSION >= 3000
  this->find_forward_algorithm(workspace_limit, deterministic);
  this->find_backward_data_algorithm(workspace_limit, deterministic);
  this->find_backward_filter_algorithm(workspace_limit, deterministic);
#else
  if (deterministic) {
    std::cout << "Note that cuDNN version 2 and earlier do not provide means "
                 "to select convolution algorithms based on deterministic "
                 "behaviour as requested by the NNABLA_CUDNN_DETERMINISTIC "
                 "environment variable."
              << std::endl;
  }
  this->get_forward_algorithm(workspace_limit);
  this->get_backward_data_algorithm(workspace_limit);
  this->get_backward_filter_algorithm(workspace_limit);
#endif
#if 0
  printf("fwd_algo %d uses %10lu bytes\n"
         "bwd_data_algo %d uses %10lu bytes\n"
         "bwd_filter_algo %d uses %10lu bytes\n",
         this->fwd_algo, this->fwd_workspace_size,
         this->bwd_data_algo, this->bwd_data_workspace_size,
         this->bwd_filter_algo, this->bwd_filter_workspace_size);
#endif
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
            "Invalid value: NNABLA_CUDNN_WORKSPACE_LIMIT=%s. Integer required.",
            e);
      }
    }
    called = true;
  }
  return workspace_limit_;
}

bool CudnnHandleManager::get_deterministic_option() {
  static bool called = false;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (!called) {
    // Read NNABLA_CUDNN_DETERMINISTIC preference from
    // environment. Default is `false` if the environment variable is
    // not present. If present, then any value other than `0`
    // (including an empty string) implies `true`.
    const char *e = std::getenv("NNABLA_CUDNN_DETERMINISTIC");
    if (!e) {
      this->deterministic_option_ = false;
    } else {
      try {
        this->deterministic_option_ = bool(std::stoi(e) != 0);
      } catch (std::exception &exc) {
        this->deterministic_option_ = true;
      }
    }
    called = true;
  }
  return deterministic_option_;
}

void CudnnHandleManager::set_deterministic_option(bool value) {
  this->deterministic_option_ = value;
}

void CudnnHandleManager::set_workspace_limit_in_bytes(int bytes) {
  workspace_limit_ = bytes;
}

NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, CudnnHandleManager);
}
