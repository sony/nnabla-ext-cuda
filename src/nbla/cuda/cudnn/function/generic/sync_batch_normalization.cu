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
#include <nbla/cuda/cudnn/function/sync_batch_normalization.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/cuda/function/kernel/batch_normalization.cuh>
#include <nbla/cuda/limits.hpp>

namespace nbla {

#define DRV_BN_T() get_dtype_by_cudnn_data_type(derived_bn_dtype_)

template <typename T>
void SyncBatchNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                    const Variables &outputs) {
  this->batch_norm_cudnn_.setup(inputs, outputs);

  SyncBatchNormalizationCuda<T>::setup_impl(inputs, outputs);

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
__global__ void forward_running_var_to_sq_mean_kernel(const int size, T *v,
                                                      const T *m, int n) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    v[i] = (n - 1.0) / n * v[i];
    v[i] = v[i] + std::pow(m[i], static_cast<T>(2));
  }
}

template <class T>
void SyncBatchNormalizationCudaCudnn<T>::forward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const bool update_inputs) {
  SyncBatchNormalizationCuda<T>::forward_impl_batch(inputs, outputs,
                                                    update_inputs);
  return;
  // TODO:
  /*
    In most combination among versions of cuda, cudnn, and nccl, the following
    code does not works, i.e., undefined behaviour or stochastic behavior when
    calling the forward of a validation graph earlier than that of a training
    graph.

    There is two possible workarounds:

    1. Call the forward of a training graph first,
    2. export NNABLA_CUDNN_WORKSPACE_LIMIT=0 to use basically 0-th algorithm.

    However, to be conservative, call the parent method.
   */

  // Check whether it outputs batch mean and var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  if (outputs.size() == 3) {
    batch_mean = outputs[1];
    batch_var = outputs[2];
  }

  // Inputs
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);
  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  // Output
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  void *m_cudnn =
      batch_mean->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch mean
  void *v_cudnn =
      batch_var->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch var

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);

  // Get batch mean/var and update running mean/var

  // Use the forward function of the batch-normalization to calculate
  // batch-mean/var
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, mode_, &a, &b, input_desc_, x, output_desc_, y,
      bn_scale_bias_mean_var_desc_, gamma, beta,
      1 /* Use batch-mean/var as running-mean/var */, m_cudnn, v_cudnn, epsilon,
      nullptr, nullptr));

  // Convert variance to squared mean
  Tw *m = batch_mean->cast_data_and_get_pointer<Tw>(this->ctx_); // batch mean
  Tw *v = batch_var->cast_data_and_get_pointer<Tw>(this->ctx_);  // batch var
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_running_var_to_sq_mean_kernel,
                                 batch_mean->size(), v, m,
                                 this->size0_ * this->size2_);

  // Sync between other processes
  this->comm_->all_reduce({batch_mean->data(), batch_var->data()}, false, false,
                          this->group_);

  m = batch_mean->cast_data_and_get_pointer<Tw>(this->ctx_); // batch mean
  v = batch_var->cast_data_and_get_pointer<Tw>(this->ctx_);  // batch var
  Tw *rm = !update_inputs ? nullptr : inputs[3]->cast_data_and_get_pointer<Tw>(
                                          this->ctx_); // running mean
  Tw *rv = !update_inputs ? nullptr : inputs[4]->cast_data_and_get_pointer<Tw>(
                                          this->ctx_); // running var
  // Calculate running mean and var
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_running_mean_var_kernel,
                                 /* Input */
                                 this->size1_, this->size0_ * this->size2_,
                                 this->num_processes_, this->decay_rate_, m, v,
                                 /* Output */
                                 rm, rv);

  // Output
  m_cudnn =
      batch_mean->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch mean
  v_cudnn =
      batch_var->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // batch var

  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      cudnn_handle_, mode_, &a, &b, input_desc_, x, output_desc_, y,
      bn_scale_bias_mean_var_desc_, gamma, beta, m_cudnn, v_cudnn, epsilon));
}

template <class T>
void SyncBatchNormalizationCudaCudnn<T>::forward_impl_global(
    const Variables &inputs, const Variables &outputs) {
  this->batch_norm_cudnn_.forward(inputs, outputs);
}
}
