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
#if CUDNN_VERSION < 7000
#error cuDNN 6.x is no longer supported. Contributions to make cuDNN 6.x compatible are welcome.
#endif

#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/singleton_manager.hpp>

#include <nbla/cuda/half.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace nbla {

using std::map;
using std::shared_ptr;
using std::unordered_map;
using std::hash;

/** Convert template type to cudnnDataType_t
*/
template <class T> class cudnn_data_type;

template <> class cudnn_data_type<float> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_FLOAT; }
};
template <> class cudnn_data_type<double> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_DOUBLE; }
};
template <> class cudnn_data_type<half> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_HALF; }
};
template <> class cudnn_data_type<Half> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_HALF; }
};
template <> class cudnn_data_type<HalfCuda> {
public:
  static cudnnDataType_t type() { return CUDNN_DATA_HALF; }
};

/** Convert cuDNN enum dtype to NNabla enum dtype.
 */
inline dtypes get_dtype_by_cudnn_data_type(cudnnDataType_t dtype) {
  switch (dtype) {
#define _T(TYPE, RET_TYPE)                                                     \
  case CUDNN_DATA_##TYPE:                                                      \
    return dtypes::RET_TYPE;
    _T(FLOAT, FLOAT);
    _T(DOUBLE, FLOAT);
    _T(HALF, HALF);
    _T(INT8, BYTE);
#if CUDNN_VERSION >= 7100
    _T(UINT8, UBYTE);
#endif
    _T(INT32, INT);
// TODO: INT8x4, UINT8x4
#undef _T
  default:
    NBLA_ERROR(error_code::value, "Unknown value of cudnnDataType_t. INT8x4 "
                                  "and UINT8x4 are not supported yet.");
  }
}

/** Return scalar value used in alpha and beta in cuDNN APIs.

    This implementation is based on the fact that cuDNN algorithm APIs take
   alpha and beta as float when storage data type is half.
 */
template <class T>
typename CudaTypeForceFloat<T>::type get_cudnn_scalar_arg(float val) {
  return val;
}

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

According to the doc above, cudnnSetTensorNdDescriptor does not support a tensor
less than 4 dimensions. This wrapper function adds unused dimensions with a
value of 1 at last.

@param force_ndim
 */
void cudnn_set_tensor_nd_descriptor_force_dim(
    cudnnTensorDescriptor_t &desc, cudnnDataType_t dtype, vector<int> dims,
    size_t force_ndim = 4, bool channel_last = false, bool expand_left = false);

template <typename T>
inline void cudnn_set_tensor_descriptor(cudnnTensorDescriptor_t desc,
                                        std::vector<int> shape) {
  if (shape.size() <= 4) {
    shape.insert(shape.end(), 4 - shape.size(), 1);
    NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc, CUDNN_TENSOR_NCHW, cudnn_data_type<T>::type(), shape.at(0),
        shape.at(1), shape.at(2), shape.at(3)));
  } else {
    // Note: Can use ndi::strides from <nbla/utils/nd_index.hpp> when
    // branch feature/pad_with_reflection is merged to master.
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(),
                     std::multiplies<int>());
    NBLA_CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        desc, cudnn_data_type<T>::type(), static_cast<int>(shape.size()),
        shape.data(), strides.data()));
  }
}

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
  bool channel_last;    ///< Channels at last dimension (NHWC).
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
      hash_combine(h, x.group);
      hash_combine(h, x.channel_last);
      for (int d = 0; d < x.ndim; d++) {
        hash_combine(h, x.sample[d]);
        hash_combine(h, x.kernel[d]);
        hash_combine(h, x.pad[d]);
        hash_combine(h, x.stride[d]);
        hash_combine(h, x.dilation[d]);
      }
      return h;
    }
  };
};

std::ostream &operator<<(std::ostream &os, const CudnnConvDesc &desc);

/** CUDNN Convolution descriptor wrapper.
 */
struct CudnnConvolutionDescriptor {
  cudnnConvolutionDescriptor_t desc;
  CudnnConvolutionDescriptor();
  ~CudnnConvolutionDescriptor();
};

/**
  CUDNN Pooling descriptor wrapper.
 */
struct CudnnPoolingDescriptor {
  cudnnPoolingDescriptor_t desc;
  CudnnPoolingDescriptor();
  ~CudnnPoolingDescriptor();
};

/**
 * CUDNN tensor descriptor wrapper.
 */
struct CudnnTensorDescriptor {
  cudnnTensorDescriptor_t desc;
  CudnnTensorDescriptor();
  ~CudnnTensorDescriptor();
};

/**
   CUDNN activation descriptor wrapper.
 */
struct CudnnActivationDescriptor {
  cudnnActivationDescriptor_t desc;
  CudnnActivationDescriptor();
  ~CudnnActivationDescriptor();
};

/**
   Common CUDNN pooling function wrapper.
 */
class CudnnPooling {
  CudnnTensorDescriptor input_desc_;
  CudnnTensorDescriptor output_desc_;
  CudnnPoolingDescriptor pooling_desc_;
  int device_;

public:
  typedef shared_ptr<CudnnPooling> Ptr;
  static Ptr create(const vector<int> &inshape, const vector<int> &kernel,
                    const vector<int> &stride, bool ignore_border,
                    const vector<int> &pad, bool channel_last,
                    cudnnPoolingMode_t mode, cudnnDataType_t dtype, int device);

  CudnnPooling(const vector<int> &inshape, const vector<int> &kernel,
               const vector<int> &stride, bool ignore_border,
               const vector<int> &pad, bool channel_last,
               cudnnPoolingMode_t mode, cudnnDataType_t dtype, int device);
  void forward(const void *alpha, const void *x, const void *beta,
               void *y) const;

  void backward(const void *alpha, const void *y, const void *dy, const void *x,
                const void *beta, void *dx) const;
};

/**
   CUDNN softmax function wrapper
 */
class CudnnSoftmax {
  CudnnTensorDescriptor input_desc_;
  CudnnTensorDescriptor output_desc_;
  cudnnSoftmaxAlgorithm_t algo_;
  int device_;

public:
  typedef shared_ptr<CudnnSoftmax> Ptr;
  CudnnSoftmax(const Shape_t &inshape, int axis, cudnnSoftmaxAlgorithm_t algo,
               cudnnDataType_t dtype, int device);
  static Ptr create(const Shape_t &inshape, int axis,
                    cudnnSoftmaxAlgorithm_t algo, cudnnDataType_t dtype,
                    int device);
  void forward(const void *alpha, const void *x, const void *beta,
               void *y) const;
  void backward(const void *alpha, const void *y, const void *dy,
                const void *beta, void *dx) const;
};

/** cuDNN Convolution resource cache.
 */
struct NBLA_CUDA_API CudnnConvResource {
  int device;                                 ///< Device ID.
  cudnnTensorDescriptor_t x_desc;             ///< Input desc.
  cudnnTensorDescriptor_t y_desc;             ///< Output desc.
  cudnnTensorDescriptor_t b_desc;             ///< Bias desc.
  cudnnTensorDescriptor_t b_desc_deconv;      ///< Bias desc for deconvolution.
  cudnnFilterDescriptor_t w_desc;             ///< Weight desc.
  CudnnConvolutionDescriptor conv_desc;       ///< Conv desc.
  CudnnConvolutionDescriptor conv_dgrad_desc; ///< Conv backward data desc.
  CudnnConvolutionDescriptor conv_wgrad_desc; ///< Conv backward filter desc.
  cudnnConvolutionFwdAlgo_t fwd_algo;         ///< Best forward algorithm found.
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
  void find_forward_algorithm(int workspace_limit, bool deterministic);
  void find_backward_data_algorithm(int workspace_limit, bool deterministic);
  void find_backward_filter_algorithm(int workspace_limit, bool deterministic);
  void get_forward_algorithm(int workspace_limit);
  void get_backward_data_algorithm(int workspace_limit);
  void get_backward_filter_algorithm(int workspace_limit);
  void find_best_algorithms();
};

/**
Singleton class for storing cudnn handle for CUDA CUDNN Computation.
*/
class NBLA_CUDA_API CudnnHandleManager {
public:
  ~CudnnHandleManager();

  /**
     Get cuDNN handle for device.
   */
  cudnnHandle_t handle(int device = -1, cudaStream_t stream = 0);

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

  /* Get option for choosing deterministic algorithms.

     True requests the use of deterministic algorithms.

     @note The default value is false. The default value is overwritten if an
           environment variable NNABLA_CUDNN_DETERMINISTIC is specified.
   */
  bool get_deterministic_option();

  /* Set option for choosing deterministic algorithms.

     True requests the use of deterministic algorithms.

     @param[in] Option value.
   */
  void set_deterministic_option(bool value);

protected:
  unordered_map<int, unordered_map<cudaStream_t, shared_ptr<cudnnHandle_t>>>
      handles_;
  int workspace_limit_{0};           ///< Workspace limit in bytes.
  bool deterministic_option_{false}; ///< Choose deterministic algorithms

private:
  friend SingletonManager;
  CudnnHandleManager();
  DISABLE_COPY_AND_ASSIGN(CudnnHandleManager);
};
}
#endif
