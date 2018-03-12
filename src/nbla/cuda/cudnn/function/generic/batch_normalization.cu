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

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/batch_normalization.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

namespace nbla {

#define DRV_BN_T() get_dtype_by_cudnn_data_type(derived_bn_dtype_)

template <typename T>
void BatchNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                const Variables &outputs) {
  BatchNormalizationCuda<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  int N = this->size0_;
  int C = this->size1_;
  int H = this->size2_;
  int W = 1;
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc_,
                                                 input_desc_, mode_));
  int n, c, h, w, sn, sc, sh, sw; // garbage
  NBLA_CUDNN_CHECK(cudnnGetTensor4dDescriptor(bn_scale_bias_mean_var_desc_,
                                              &derived_bn_dtype_, &n, &c, &h,
                                              &w, &sn, &sc, &sh, &sw));
}

template <class T>
void BatchNormalizationCudaCudnn<T>::forward_impl(const Variables &inputs,
                                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->batch_stat_) { // Training mode.
    forward_impl_batch(inputs, outputs);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <class T>
void BatchNormalizationCudaCudnn<T>::forward_impl_batch(
    const Variables &inputs, const Variables &outputs) {

  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    // [WORKAROUND]
    // To save mean and variance is not supported with cuDNN.
    // Because cuDNN's forward training interface is different from NNabla's
    // one.
    // So Fall back to CUDA implementation if outputs.size() == 3
    // TODO: Change saved variance to inverse variance like cuDNN
    BatchNormalizationCuda<T>::forward_impl_batch(inputs, outputs);
    return;
  }
  // Inputs
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  // Output
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_);
  void *m =
      batch_mean->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch mean
  void *v =
      batch_var->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch var
  // Inputs/Outputs
  void *rm = inputs[3]
                 ->data()
                 ->cast(DRV_BN_T(), this->ctx_)
                 ->pointer(); // running mean
  void *rv =
      inputs[4]->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // running var
  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, mode_, &a, &b, input_desc_, x, output_desc_, y,
      bn_scale_bias_mean_var_desc_, gamma, beta, 1 - this->decay_rate_, rm, rv,
      epsilon, m, v));
}

template <class T>
void BatchNormalizationCudaCudnn<T>::forward_impl_global(
    const Variables &inputs, const Variables &outputs) {
  // Inputs
  // while(1);
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *rm = inputs[3]
                       ->data()
                       ->get(DRV_BN_T(), this->ctx_)
                       ->const_pointer(); // running mean
  const void *rv = inputs[4]
                       ->data()
                       ->get(DRV_BN_T(), this->ctx_)
                       ->const_pointer(); // running var
  // Output
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_);

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      cudnn_handle_, mode_, &a, &b, input_desc_, x, output_desc_, y,
      bn_scale_bias_mean_var_desc_, gamma, beta, rm, rv, epsilon));
}

template <class T>
void BatchNormalizationCudaCudnn<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {

  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->batch_stat_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    NBLA_ERROR(error_code::not_implemented, "");
  }
}

template <class T>
void BatchNormalizationCudaCudnn<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    // [WORKAROUND]
    // To use saved mean and variance and to propagate mean and variance
    // gradient are not supported with cuDNN.
    // Because cuDNN's backward interface is different from NNabla's one.
    // So Fall back to CUDA implementation if outputs.size() == 3
    // TODO: Change saved variance to inverse variance like cuDNN
    BatchNormalizationCuda<T>::backward_impl_batch(inputs, outputs,
                                                   propagate_down, accum);
    return;
  }
  // Commont inputs wrt. gradient.
  const Tw *dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  const void *m =
      batch_mean->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *v =
      batch_var->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);

  if (propagate_down[0] || propagate_down[1] || propagate_down[2]) {
    auto a_data = get_cudnn_scalar_arg<T>(propagate_down[0] ? 1 : 0);
    auto b_data =
        get_cudnn_scalar_arg<T>(accum[0] && propagate_down[0] ? 1 : 0);
    auto a_param =
        get_cudnn_scalar_arg<T>(propagate_down[1] || propagate_down[2] ? 1 : 0);
    auto b_param = a_param;
    if (!(accum[1] || accum[2])) {
      b_param = 0;
    } else {
      if (!accum[1])
        inputs[1]->grad()->zero();
      if (!accum[2])
        inputs[2]->grad()->zero();
    }

    size_t workspace_size = 0;
    if (!propagate_down[0]) {
      workspace_size = std::max(workspace_size,
                                inputs[0]->size() * sizeof_dtype(DRV_BN_T()));
    }
    if (!propagate_down[1] || !propagate_down[2]) {
      workspace_size = std::max(workspace_size,
                                inputs[1]->size() * sizeof_dtype(DRV_BN_T()));
    }
    void *tmp_buf = nullptr;
    shared_ptr<CudaCachedArray> mem_workspace(
        workspace_size
            ? new CudaCachedArray(workspace_size, dtypes::BYTE, this->ctx_)
            : nullptr);
    if (workspace_size) {
      tmp_buf = mem_workspace->pointer();
    }

    Tw *dx = propagate_down[0]
                 ? inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_)
                 : (Tw *)tmp_buf;
    const void *gamma =
        inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
    void *db = propagate_down[1]
                   ? inputs[1]->grad()->cast(DRV_BN_T(), this->ctx_)->pointer()
                   : tmp_buf;
    void *dg = propagate_down[2]
                   ? inputs[2]->grad()->cast(DRV_BN_T(), this->ctx_)->pointer()
                   : tmp_buf;
    NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        cudnn_handle_, mode_, &a_data, &b_data, &a_param, &b_param, input_desc_,
        x, output_desc_, dy, input_desc_, dx, bn_scale_bias_mean_var_desc_,
        gamma, dg, db, epsilon, m, v));
  }
}
}
