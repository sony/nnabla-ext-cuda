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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/batch_normalization.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/cuda/limits.hpp>

namespace nbla {

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
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);

  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T *m = batch_mean->cast_data_and_get_pointer<T>(this->ctx_); // batch mean
  T *v = batch_var->cast_data_and_get_pointer<T>(this->ctx_);  // batch varf
  // Inputs/Outputs
  T *rm = inputs[3]->cast_data_and_get_pointer<T>(this->ctx_); // running mean
  T *rv = inputs[4]->cast_data_and_get_pointer<T>(this->ctx_); // running var
  T a = 1;
  T b = 0;
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
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *beta = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
  const T *rm = inputs[3]->get_data_pointer<T>(this->ctx_); // running mean
  const T *rv = inputs[4]->get_data_pointer<T>(this->ctx_); // running var
  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  T a = 1;
  T b = 0;
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
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *m = batch_mean->get_data_pointer<T>(this->ctx_);
  const T *v = batch_var->get_data_pointer<T>(this->ctx_);
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);

  if (propagate_down[0] && (propagate_down[1] || propagate_down[2])) {
    T a = 1;
    T b_data = accum[0] ? 1 : 0;
    T b_param = 1;
    if (!(accum[1] || accum[2])) {
      b_param = 0;
    } else {
      if (!accum[1])
        inputs[1]->grad()->zero();
      if (!accum[2])
        inputs[2]->grad()->zero();
    }

    T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    const T *gamma = inputs[2]->get_data_pointer<T>(this->ctx_);
    NBLA_CHECK(propagate_down[1] && propagate_down[2], error_code::value,
               "'need_grad' of beta and gamma must be the same.");
    T *db = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    T *dg = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_);
    NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        cudnn_handle_, mode_, &a, &b_data, &a, &b_param, input_desc_, x,
        output_desc_, dy, input_desc_, dx, bn_scale_bias_mean_var_desc_, gamma,
        dg, db, epsilon, m, v));
  }
}

template class BatchNormalizationCudaCudnn<float>;
}
