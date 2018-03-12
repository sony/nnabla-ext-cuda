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

/** Utilities for CUDA CUDNN
*/
#ifndef __NBLA_CUDA_CUDNN_HPP__
#define __NBLA_CUDA_CUDNN_HPP__

#include <cudnn.h>

#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/singleton_manager.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

namespace nbla {

using std::map;
using std::shared_ptr;
using std::unordered_map;
using std::hash;

template <class T> class cudnn_data_type;

template <> class cudnn_data_type<float> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_FLOAT; }
};
template <> class cudnn_data_type<double> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_DOUBLE; }
};
template <> class cudnn_data_type<unsigned short> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_HALF; }
};

inline string cudnn_status_to_string(cudnnStatus_t status) {
#define CASE_CUDNN_STATUS(NAME)                                                \
  case CUDNN_STATUS_##NAME:                                                    \
    return #NAME;

  switch (status) {
    CASE_CUDNN_STATUS(SUCCESS);
    CASE_CUDNN_STATUS(NOT_INITIALIZED);
    CASE_CUDNN_STATUS(ALLOC_FAILED);
    CASE_CUDNN_STATUS(BAD_PARAM);
    CASE_CUDNN_STATUS(INTERNAL_ERROR);
    CASE_CUDNN_STATUS(INVALID_VALUE);
    CASE_CUDNN_STATUS(ARCH_MISMATCH);
    CASE_CUDNN_STATUS(MAPPING_ERROR);
    CASE_CUDNN_STATUS(EXECUTION_FAILED);
    CASE_CUDNN_STATUS(NOT_SUPPORTED);
    CASE_CUDNN_STATUS(LICENSE_ERROR);
#if CUDNN_VERSION >= 6000
    CASE_CUDNN_STATUS(RUNTIME_PREREQUISITE_MISSING);
#endif
#if CUDNN_VERSION >= 7000
    CASE_CUDNN_STATUS(RUNTIME_IN_PROGRESS);
    CASE_CUDNN_STATUS(RUNTIME_FP_OVERFLOW);
#endif
  }
  return "UNKNOWN";
#undef CASE_CUDNN_STATUS
}

#define NBLA_CUDNN_CHECK(condition)                                            \
  {                                                                            \
    cudnnStatus_t status = condition;                                          \
    NBLA_CHECK(status == CUDNN_STATUS_SUCCESS, error_code::target_specific,    \
               cudnn_status_to_string(status));                                \
  }

/** Wrapper function of cudnnSetTensorNdDescriptor with ensuring least dims.

http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSetTensorNdDescriptor

According to the doc above, cudnnSetTensorNdDescriptor does not suport a tensor
less than 4 dimensions. This wrapper function adds unused dimensions with a
value of 1 at last.

@param force_ndim
 */
void cudnn_set_tensor_nd_descriptor_force_dim(cudnnTensorDescriptor_t &desc,
                                              cudnnDataType_t dtype,
                                              vector<int> dims,
                                              vector<int> strides,
                                              int force_ndim = 4);

/** cuDNN Convolution Descriptor used as a key to find previously used
 * (cached) config.
*/
struct NBLA_CUDA_API CudnnConvDesc {
  int ndim;              ///< Dimension of spatial samples.
  int device;            ///< Device ID.
  cudnnDataType_t dtype; ///< Data type.
  cudnnConvolutionMode_t
      mode;             ///< CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION;
  int n;                ///< Batch size.
  int c;                ///< Channels of input.
  int o;                ///< Channels of output.
  int group;            ///< Number of groups.
  vector<int> sample;   ///< Sample size of each dimension.
  vector<int> kernel;   ///< Kernel size of each dimension.
  vector<int> pad;      ///< Padding size of each dimension.
  vector<int> stride;   ///< Stride size of each dimension.
  vector<int> dilation; ///< Dilation size of each dimension.

  /// Operator == compares all elements.
  bool operator==(const CudnnConvDesc &right) const;

  /** Custom hash function for CudnnConvDesc.
   */
  class Hash {
  public:
    std::size_t operator()(const CudnnConvDesc &x) const {
      size_t h = hash<int>{}(x.device);
      hash_combine(h, static_cast<int>(x.dtype));
      hash_combine(h, static_cast<int>(x.mode));
      hash_combine(h, x.n);
      hash_combine(h, x.c);
      hash_combine(h, x.o);
      for (int d = 0; d < x.ndim; d++) {
        hash_combine(h, x.sample[d]);
        hash_combine(h, x.kernel[d]);
        hash_combine(h, x.pad[d]);
        hash_combine(h, x.stride[d]);
        hash_combine(h, x.dilation[d]);
      }
      hash_combine(h, x.group);
      return h;
    }
  };
};

std::ostream &operator<<(std::ostream &os, const CudnnConvDesc &desc);

/** cuDNN Convolution resource cache.
 */
struct NBLA_CUDA_API CudnnConvResource {
  int device;                             ///< Device ID.
  cudnnTensorDescriptor_t x_desc;         ///< Input desc.
  cudnnTensorDescriptor_t y_desc;         ///< Output desc.
  cudnnTensorDescriptor_t b_desc;         ///< Bias desc.
  cudnnTensorDescriptor_t b_desc_deconv;  ///< Bias desc for deconvolution.
  cudnnFilterDescriptor_t w_desc;         ///< Wegiht desc.
  cudnnConvolutionDescriptor_t conv_desc; ///< Conv desc.
  cudnnConvolutionFwdAlgo_t fwd_algo;     ///< Best forward algorithm found.
  cudnnConvolutionBwdFilterAlgo_t
      bwd_filter_algo; ///< Best Backward filter algorithm found.
  cudnnConvolutionBwdDataAlgo_t
      bwd_data_algo;                ///< Best backward data algorithm found.
  size_t fwd_workspace_size;        ///< Forward workspace size.
  size_t bwd_filter_workspace_size; ///< Backward filter workspace size.
  size_t bwd_data_workspace_size;   ///< Backward data workspace size.

  CudnnConvResource(const CudnnConvDesc &desc);
  ~CudnnConvResource();

  /** Get maximum workspace size.
   */
  size_t workspace_size() const;

private:
  void find_best_algorithms();
  void find_best_algorithms_no_limit();
  void find_best_algorithms_no_workspace();
  void find_best_algorithms_limit(int limit);
};

/**
Singleton class for storing cudnn handle for CUDA CUDNN Computation.
*/
class NBLA_CUDA_API CudnnHandleManager {
public:
  ~CudnnHandleManager();

  /**
     Get cuDNN handle for devive.
   */
  cudnnHandle_t handle(int device = -1);

  /** Hash map for CudnnConvResource.
   */
  unordered_map<CudnnConvDesc, shared_ptr<CudnnConvResource>,
                typename CudnnConvDesc::Hash>
      conv_resource;

  /** Get a workspace limit.

      The negative value means no limit of workspace size.

      @note The default value is -1. The default value is overwritten if an
            environment variable NNABLA_CUDNN_WORKSPACE_LIMIT is specified.
   */
  int get_workspace_limit_in_bytes();

  /** Set a workspace limit.

      The negative value means no limit of workspace size.

      @param[in] Limit in bytes.
   */
  void set_workspace_limit_in_bytes(int bytes);

protected:
  map<int, cudnnHandle_t> handles_;
  int workspace_limit_{0}; ///< Workspace limit in bytes.

private:
  friend SingletonManager;
  CudnnHandleManager();
  DISABLE_COPY_AND_ASSIGN(CudnnHandleManager);
};
}
#endif
