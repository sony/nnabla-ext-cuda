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

/** Batch Normalization
 */
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_BATCHNORM_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_BATCHNORM_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>

#include <vector>

using std::vector;

namespace nbla {

template <typename T>
class BatchNormalizationCudaCudnn : public BatchNormalizationCuda<T> {
protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  CudnnTensorDescriptor input_desc_, output_desc_;
  CudnnTensorDescriptor bn_scale_bias_mean_var_desc_;
  cudnnDataType_t derived_bn_dtype_;
  cudnnBatchNormMode_t mode_;
#if CUDNN_VERSION >= 7400
  bool can_use_bn_ex_{false};
  CudnnActivationDescriptor act_desc_;
  NdArrayPtr reserve_;
  cudnnBatchNormOps_t ops_{CUDNN_BATCHNORM_OPS_BN};
  size_t forward_workspace_size_{0};
  size_t backward_workspace_size_{0};
  size_t reserve_size_{0};
#endif

public:
  typedef typename CudaType<T>::type Tw;

  BatchNormalizationCudaCudnn(const Context &ctx, const vector<int> axes,
                              float decay_rate, float eps, bool batch_stat)
      : BatchNormalizationCuda<T>(ctx, axes, decay_rate, eps, batch_stat),
        device_(std::stoi(ctx.device_id)) {
#if CUDNN_VERSION < 5000
    std::cout << "Falling back to BatchNormalizationCuda since BN does not "
                 "exist in CUDNN_VERSION < 5000."
              << std::endl; // TODO: warn.
    this->fall_back_func_.reset(
        new BatchNormalizationCuda<T>(ctx, axes, decay_rate, eps, batch_stat));
#else
    NBLA_CHECK(eps >= (float)CUDNN_BN_MIN_EPSILON, error_code::value,
               "eps must be greater than or equal to CUDNN_BN_MIN_EPSILON. "
               "eps=%g, CUDNN_BN_MIN_EPSILON=%g",
               eps, CUDNN_BN_MIN_EPSILON);
#endif
  }
  virtual ~BatchNormalizationCudaCudnn() {}
  virtual string name() { return "BatchNormalizationCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual void forward_impl_batch(const Variables &inputs,
                                  const Variables &outputs);
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs);
  virtual void backward_impl_batch(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum);
};
}
#endif
