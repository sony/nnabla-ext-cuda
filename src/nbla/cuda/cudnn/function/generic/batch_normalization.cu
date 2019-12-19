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

#include <type_traits>

namespace nbla {

#define DRV_BN_T() get_dtype_by_cudnn_data_type(derived_bn_dtype_)

template <typename T>
void BatchNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                const Variables &outputs) {
  if (outputs.size() == 3) {
    // [WORKAROUND]
    // To use saved mean and variance and to propagate mean and variance
    // gradient are not supported with cuDNN.
    // Because cuDNN's backward interface is different from NNabla's one.
    // So Fall back to CUDA implementation if outputs.size() == 3
    // TODO: Change saved variance to inverse variance like cuDNN
    this->fall_back_func_ = make_shared<BatchNormalizationCuda<T>>(
        this->ctx_, this->axes_, this->decay_rate_, this->eps_,
        this->batch_stat_);
    this->fall_back_func_->setup(inputs, outputs);
    return;
  }
  BatchNormalizationCuda<T>::setup_impl(inputs, outputs);
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  NBLA_CHECK(this->axes_.size() == 1, error_code::value,
             "Axes on a single dimension is only supported.");
  int N = this->size0_;
  int C = this->size1_;
  int H = this->size2_;
  int W = 1;
  mode_ = CUDNN_BATCHNORM_SPATIAL;
  // Channel last is restricted for spatial input
  bool channel_last = this->axes_[0] == inputs[0]->ndim() - 1;
  if (inputs[0]->ndim() == 2) { // typical 1-d affine output with shape (N, C)
    mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc_.desc, CUDNN_TENSOR_NHWC,
                                   cudnn_data_type<T>::type(), N, C, H, W));
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(output_desc_.desc, CUDNN_TENSOR_NHWC,
                                   cudnn_data_type<T>::type(), N, C, H, W));
  } else if (channel_last) {
    // To prevent NOT SUPPORTED error in CUDNNN, N and H are recalculated.
    // (Large N is not allowed.)
    N = inputs[0]->shape()[0];
    H = inputs[0]->size() / (N * C);
    if (this->batch_stat_) {
      // cudnnBatchNormalizationForwardInference does not support this mode.
      mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    }
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc_.desc, CUDNN_TENSOR_NHWC,
                                   cudnn_data_type<T>::type(), N, C, H, W));
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(output_desc_.desc, CUDNN_TENSOR_NHWC,
                                   cudnn_data_type<T>::type(), N, C, H, W));
  } else {
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc_.desc, CUDNN_TENSOR_NCHW,
                                   cudnn_data_type<T>::type(), N, C, H, W));
    NBLA_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(output_desc_.desc, CUDNN_TENSOR_NCHW,
                                   cudnn_data_type<T>::type(), N, C, H, W));
  }

  // Get BN data type.
  NBLA_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
      bn_scale_bias_mean_var_desc_.desc, input_desc_.desc, mode_));
  int n, c, h, w, sn, sc, sh, sw; // garbage
  NBLA_CUDNN_CHECK(cudnnGetTensor4dDescriptor(bn_scale_bias_mean_var_desc_.desc,
                                              &derived_bn_dtype_, &n, &c, &h,
                                              &w, &sn, &sc, &sh, &sw));
#if CUDNN_VERSION >= 7400
  // Check if the confition we can use faster BN.
  can_use_bn_ex_ =
      channel_last && std::is_same<Tw, nbla::HalfCuda>::value && C % 4 == 0;
  can_use_bn_ex_ &= this->batch_stat_;
  if (can_use_bn_ex_) {
    NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        this->cudnn_handle_, this->mode_, this->ops_,
        this->input_desc_.desc,  /* x desc */
        nullptr,                 /* z desc */
        this->output_desc_.desc, /* y desc */
        this->bn_scale_bias_mean_var_desc_.desc, nullptr,
        &forward_workspace_size_));

    NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        this->cudnn_handle_, this->mode_, this->ops_, this->act_desc_.desc,
        this->input_desc_.desc, &reserve_size_));

    NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        this->cudnn_handle_, this->mode_, this->ops_,
        this->input_desc_.desc,  /* x desc */
        this->output_desc_.desc, /* y desc */
        this->output_desc_.desc, /* dy desc */
        this->input_desc_.desc,  /*dz desc*/
        this->input_desc_.desc,  /* dx desc */
        this->bn_scale_bias_mean_var_desc_.desc, this->act_desc_.desc,
        &backward_workspace_size_));
  }
#endif
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

  // Inputs
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);

  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  // Output
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  void *m = batch_mean->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch mean
  void *v = batch_var->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch var
  // Inputs/Outputs
  void *rm = inputs[3]
                 ->data()
                 ->cast(DRV_BN_T(), this->ctx_)
                 ->pointer(); // running mean
  void *rv =
      inputs[4]->data()->cast(DRV_BN_T(), this->ctx_)->pointer(); // running var

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
#if CUDNN_VERSION >= 7400
  if (can_use_bn_ex_) {
    // Get buffers.
    NdArray workspace(Shape_t{(Size_t)forward_workspace_size_});
    reserve_ = make_shared<NdArray>(Shape_t{(Size_t)reserve_size_});
    void *workspace_ptr =
        workspace.cast(DRV_BN_T(), this->ctx_, true)->pointer();
    void *reserve_ptr = reserve_->cast(DRV_BN_T(), this->ctx_, true)->pointer();
    // Execute forward.
    NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
        this->cudnn_handle_, this->mode_, this->ops_, &a, &b, input_desc_.desc,
        x,                    /* x */
        nullptr, nullptr,     /* z */
        output_desc_.desc, y, /* y */
        this->bn_scale_bias_mean_var_desc_.desc, gamma, beta,
        1 - this->decay_rate_, rm, rv, eps, m, v,
        this->act_desc_.desc,    /* activation descriptor */
        workspace_ptr,           /* workspace pointer */
        forward_workspace_size_, /* workspace size */
        reserve_ptr,             /* reserve space pointer */
        reserve_size_            /* reserve space size */
        ));
    return;
  }
#endif
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, mode_, &a, &b, input_desc_.desc, x, output_desc_.desc, y,
      bn_scale_bias_mean_var_desc_.desc, gamma, beta, 1 - this->decay_rate_, rm,
      rv, eps, m, v));
}

template <class T>
void BatchNormalizationCudaCudnn<T>::forward_impl_global(
    const Variables &inputs, const Variables &outputs) {
  // Inputs
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
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
      cudnn_handle_, mode_, &a, &b, input_desc_.desc, x, output_desc_.desc, y,
      bn_scale_bias_mean_var_desc_.desc, gamma, beta, rm, rv, eps));
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
  // Common inputs wrt. gradient.
  const Tw *dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  const void *m =
      batch_mean->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *v =
      batch_var->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);

  auto a_data = get_cudnn_scalar_arg<T>(propagate_down[0] ? 1 : 0);
  auto b_data = get_cudnn_scalar_arg<T>(accum[0] && propagate_down[0] ? 1 : 0);
  auto a_param =
      get_cudnn_scalar_arg<T>(propagate_down[1] || propagate_down[2] ? 1 : 0);
  auto b_param = a_param;
  if (!(accum[1] || accum[2])) {
    b_param = 0;
  }

  size_t prop_down_workspace_size = 0;
  if (!propagate_down[0]) {
    prop_down_workspace_size = std::max(
        prop_down_workspace_size, inputs[0]->size() * sizeof_dtype(DRV_BN_T()));
  }
  if (!propagate_down[1] || !propagate_down[2]) {
    prop_down_workspace_size = std::max(
        prop_down_workspace_size, inputs[1]->size() * sizeof_dtype(DRV_BN_T()));
  }
  void *prop_down_buf = nullptr;
  shared_ptr<CudaCachedArray> prop_down_workspace(
      prop_down_workspace_size ? new CudaCachedArray(prop_down_workspace_size,
                                                     dtypes::BYTE, this->ctx_)
                               : nullptr);
  if (prop_down_workspace_size) {
    prop_down_buf = prop_down_workspace->pointer();
  }

  Tw *dx = propagate_down[0]
               ? inputs[0]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[0])
               : (Tw *)prop_down_buf;

  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  // Specify write only flag to prevent unnecessary memset.
  const bool param_diff_write = b_param == 0;
  void *db = propagate_down[1]
                 ? inputs[1]
                       ->grad()
                       ->cast(DRV_BN_T(), this->ctx_, param_diff_write)
                       ->pointer()
                 : prop_down_buf;
  void *dg = propagate_down[2]
                 ? inputs[2]
                       ->grad()
                       ->cast(DRV_BN_T(), this->ctx_, param_diff_write)
                       ->pointer()
                 : prop_down_buf;
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
#if CUDNN_VERSION >= 7400
  if (can_use_bn_ex_) {
    // Get buffers.
    NdArray workspace(Shape_t{(Size_t)backward_workspace_size_});
    NBLA_CHECK(reserve_, error_code::value, "Forward is not called.");
    void *workspace_ptr =
        workspace.cast(DRV_BN_T(), this->ctx_, true)->pointer();
    void *reserve_ptr =
        reserve_->cast(DRV_BN_T(), this->ctx_, false /* rw access */)
            ->pointer();
    // Execute backward.
    NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
        this->cudnn_handle_, this->mode_, this->ops_, &a_data, &b_data,
        &a_param, &b_param, input_desc_.desc, x, /* x */
        nullptr, nullptr,                        /* y */
        output_desc_.desc, dy,                   /* dy */
        nullptr, nullptr,                        /* dz == null */
        input_desc_.desc, dx,                    /* dx */
        this->bn_scale_bias_mean_var_desc_.desc, gamma, beta, dg, db, eps, m, v,
        this->act_desc_.desc,     /* activation descriptor */
        workspace_ptr,            /* workspace pointer */
        backward_workspace_size_, /* workspace size */
        reserve_ptr,              /* reserve space pointer */
        reserve_size_             /* reserve space size */
        ));
    // Clear reserved buffer for backward
    reserve_ = nullptr;
    return;
  }
#endif
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackward(
      cudnn_handle_, mode_, &a_data, &b_data, &a_param, &b_param,
      input_desc_.desc, x, output_desc_.desc, dy, input_desc_.desc, dx,
      bn_scale_bias_mean_var_desc_.desc, gamma, dg, db, eps, m, v));
}
} // namespace nbla
