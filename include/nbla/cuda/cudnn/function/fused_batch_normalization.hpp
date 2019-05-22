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
#pragma once

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/function/fused_batch_normalization.hpp>

#include <type_traits>
#include <vector>

using std::vector;

namespace nbla {

template <typename T>
class FusedBatchNormalizationCudaCudnn : public FusedBatchNormalization<T> {
protected:
  int device_;
#if CUDNN_VERSION >= 7400
  Variable mean_;
  Variable var_;
  cudnnHandle_t cudnn_handle_;
  CudnnTensorDescriptor input_desc_, z_desc_, output_desc_;
  CudnnTensorDescriptor bn_scale_bias_mean_var_desc_;
  cudnnDataType_t derived_bn_dtype_;
  cudnnBatchNormMode_t mode_;
  CudnnActivationDescriptor act_desc_;
  NdArrayPtr reserve_;
  cudnnBatchNormOps_t ops_{CUDNN_BATCHNORM_OPS_BN};
  size_t forward_workspace_size_{0};
  size_t backward_workspace_size_{0};
  size_t reserve_size_{0};
#endif

public:
  typedef typename CudaType<T>::type Tw;

  FusedBatchNormalizationCudaCudnn(const Context &ctx, const vector<int> axes,
                                   float decay_rate, float eps, bool batch_stat,
                                   const string &nonlinearity)
      : FusedBatchNormalization<T>(ctx, axes, decay_rate, eps, batch_stat,
                                   nonlinearity),
        device_(std::stoi(ctx.device_id)) {
#if CUDNN_VERSION >= 7400
    // Note: The below is_same test causes unreachable statement warning during
    // compiling. C++11 does not give any functionality for testing types at
    // compile-time.
    if (!std::is_same<Tw, HalfCuda>::value || !batch_stat) {
      this->fall_back_func_ = make_shared<FusedBatchNormalization<T>>(
          ctx, axes, decay_rate, eps, batch_stat, nonlinearity);
      return;
    }
    this->mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    NBLA_CHECK(nonlinearity == "relu", error_code::value,
               "Currently \"relu\" only supported.");
    NBLA_CHECK(eps >= (float)CUDNN_BN_MIN_EPSILON, error_code::value,
               "eps must be greater than or equal to CUDNN_BN_MIN_EPSILON. "
               "eps=%g, CUDNN_BN_MIN_EPSILON=%g",
               eps, CUDNN_BN_MIN_EPSILON);
    NBLA_CUDNN_CHECK(cudnnSetActivationDescriptor(this->act_desc_.desc,
                                                  CUDNN_ACTIVATION_RELU,
                                                  CUDNN_PROPAGATE_NAN, T(0)));
#endif
  }
  virtual ~FusedBatchNormalizationCudaCudnn() {}
  virtual string name() { return "FusedBatchNormalizationCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }
  // FusedBN CUDNN relies on output buffer.
  virtual bool grad_depends_output_data(int i, int o) const { return true; }

protected:
#if CUDNN_VERSION > 7400
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
#endif
};
} // namespace nbla
