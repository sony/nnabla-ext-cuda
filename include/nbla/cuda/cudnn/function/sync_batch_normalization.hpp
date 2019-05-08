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
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_SYNC_BATCHNORM_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_SYNC_BATCHNORM_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/batch_normalization.hpp>
#include <nbla/cuda/function/sync_batch_normalization.hpp>

#include <vector>

using std::vector;

namespace nbla {

template <typename T>
class SyncBatchNormalizationCudaCudnn : public SyncBatchNormalizationCuda<T> {
protected:
  int device_;
  cudnnBatchNormMode_t mode_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc_;
  cudnnDataType_t derived_bn_dtype_;
  double epsilon;

  BatchNormalizationCudaCudnn<T> batch_norm_cudnn_;

public:
  typedef typename CudaType<T>::type Tw;

  SyncBatchNormalizationCudaCudnn(const Context &ctx,
                                  const std::shared_ptr<Communicator> &comm,
                                  const std::string &group,
                                  const vector<int> axes, float decay_rate,
                                  float eps, bool batch_stat)
      : SyncBatchNormalizationCuda<T>(ctx, comm, group, axes, decay_rate, eps,
                                      batch_stat),
        device_(std::stoi(ctx.device_id)),
        batch_norm_cudnn_(ctx, axes, decay_rate, eps, batch_stat) {
#if CUDNN_VERSION < 5000
    std::cout << "Falling back to BatchNormalizationCuda since BN does not "
                 "exist in CUDNN_VERSION < 5000."
              << std::endl; // TODO: warn.
    this->fall_back_func_.reset(
        new BatchNormalizationCuda<T>(ctx, axes, decay_rate, eps, batch_stat));
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
    NBLA_CUDNN_CHECK(
        cudnnCreateTensorDescriptor(&bn_scale_bias_mean_var_desc_));
    // NOTE: epsilon should be less than CUDNN_BN_MIN_EPSILON
    epsilon = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
#endif
  }
  virtual ~SyncBatchNormalizationCudaCudnn() {
    if (this->fall_back_func_)
      return;
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
    NBLA_CUDNN_CHECK(
        cudnnDestroyTensorDescriptor(bn_scale_bias_mean_var_desc_));
  }
  virtual string name() override { return "SyncBatchNormalizationCudaCudnn"; }
  virtual vector<string> allowed_array_classes() override {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_batch(const Variables &inputs,
                                  const Variables &outputs) override;
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs) override;
};
}
#endif
