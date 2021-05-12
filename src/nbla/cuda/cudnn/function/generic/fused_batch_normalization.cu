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
#include <nbla/logger.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/add2.hpp>
#include <nbla/cuda/cudnn/function/batch_normalization.hpp>
#include <nbla/cuda/cudnn/function/fused_batch_normalization.hpp>
#include <nbla/cuda/cudnn/function/relu.hpp>
#include <nbla/cuda/limits.hpp>

// If you face any issue when executing fused BN without a residual input
// (inputs[5]), try enabling the following macro to do a workaround.
// #define WORKAROUND_FOR_BUG_OPS_BN_ACTIVATION

namespace nbla {

#if CUDNN_VERSION >= 7400

#define DRV_BN_T() get_dtype_by_cudnn_data_type(derived_bn_dtype_)

template <typename T>
void FusedBatchNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                     const Variables &outputs) {
  FusedBatchNormalization<T>::setup_impl(inputs, outputs);
  NBLA_CHECK(this->axes_.size() == 1, error_code::value,
             "Axes on a single dimension only supported.");
  bool channel_last = this->axes_[0] == inputs[0]->ndim() - 1;
  auto inshape = inputs[0]->shape();
  NBLA_CHECK(inputs[0]->ndim() >= 2, error_code::value,
             "Input dimensions must be >= 2.");
  int C = inshape[this->axes_[0]];
  int N = inshape[0];
  int H = inputs[0]->size() / (C * N);
  int W = 1;
  // Check if the confition we can use faster BN.
  bool can_use_bn_ex = channel_last && C % 4 == 0;
#if _WIN32
  // On windows, cudnnBatchNormalization*Ex with fused option raises error with
  // CUDNN_STATUS_NOT_SUPPORTED.
  // (The case when bnOps = {CUDNN_BATCHNORM_OPS_BN_ACTIVATION,
  // CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION}.)
  // Therefore, can_use_bn_ex is fored to be False and FusedBN fallbackes to the
  // composite one.
  if (can_use_bn_ex) {
    NBLA_LOG_WARN(
        "[FusedBatchNormalization] "
        "Currently cudnn doesn't support fusedBatchNormalization on windows. "
        "Fallbacks to a composite implementation.")
    can_use_bn_ex = false;
  }
#endif // _WIN32
  if (can_use_bn_ex) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, this->device_);
    if ((prop.major == 5) && (prop.minor == 3)) {
      NBLA_LOG_WARN("FusedBatchNormalization is not supported by CuDNN on "
                    "compute archtitecture 5.3 - "
                    "fallback to composite implementation.")
      can_use_bn_ex = false;
    }
  }
  if (!can_use_bn_ex || outputs.size() == 3) {
    this->fall_back_func_ = make_shared<FusedBatchNormalization<T>>(
        this->ctx_, this->axes_, this->decay_rate_, this->eps_,
        this->batch_stat_, this->nonlinearity_);
    this->fall_back_func_->setup(inputs, outputs);
    return;
  }

  mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);
  NBLA_CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(input_desc_.desc, CUDNN_TENSOR_NHWC,
                                 cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      z_desc_.desc, CUDNN_TENSOR_NHWC, cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(output_desc_.desc, CUDNN_TENSOR_NHWC,
                                 cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
      bn_scale_bias_mean_var_desc_.desc, input_desc_.desc, mode_));

  int n, c, h, w, sn, sc, sh, sw; // garbage
  NBLA_CUDNN_CHECK(cudnnGetTensor4dDescriptor(bn_scale_bias_mean_var_desc_.desc,
                                              &derived_bn_dtype_, &n, &c, &h,
                                              &w, &sn, &sc, &sh, &sw));

  // TODO: CUDNN_BATCHNORM_OPS_BN_ACTIVATION cannot pass the unit test
  this->ops_ = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
#if !defined(WORKAROUND_FOR_BUG_OPS_BN_ACTIVATION)
  if (inputs.size() != 6) {
    this->ops_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
  }
#endif

  // workspace allocation
  NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      this->cudnn_handle_, this->mode_, this->ops_,
      this->input_desc_.desc,  /* x desc */
      z_desc_.desc,            /* z desc */
      this->output_desc_.desc, /* y desc */
      this->bn_scale_bias_mean_var_desc_.desc, this->act_desc_.desc,
      &forward_workspace_size_));

  NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      this->cudnn_handle_, this->mode_, this->ops_, this->act_desc_.desc,
      this->input_desc_.desc, &reserve_size_));

  NBLA_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      this->cudnn_handle_, this->mode_, this->ops_,
      this->input_desc_.desc,  /* x desc */
      this->output_desc_.desc, /* y desc */
      this->output_desc_.desc, /* dy desc */
      this->z_desc_.desc,      /*dz desc*/
      this->input_desc_.desc,  /* dx desc */
      this->bn_scale_bias_mean_var_desc_.desc, this->act_desc_.desc,
      &backward_workspace_size_));
}

template <class T>
void FusedBatchNormalizationCudaCudnn<T>::fused_batch_norm_forward(
    const Variables &inputs, const Variables &outputs,
    const bool update_inputs) {
  NBLA_CHECK(this->batch_stat_, error_code::runtime,
             "If batch_stat is false, this function should not be called.");
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  batch_mean->reshape(inputs[1]->shape(), true);
  batch_var->reshape(inputs[2]->shape(), true);

  // Inputs
  const Tw *x = inputs[0]->get_data_pointer<Tw>(this->ctx_);

  const void *beta =
      inputs[1]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  const void *gamma =
      inputs[2]->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();

  const void *z = inputs.size() == 6
                      ? inputs[5]->get_data_pointer<Tw>(this->ctx_)
                      : nullptr;

#if defined(WORKAROUND_FOR_BUG_OPS_BN_ACTIVATION)
  NdArray z_tmp(inputs[0]->shape());
  if (z == nullptr) {
    z_tmp.zero();
    z = z_tmp.get(DRV_BN_T(), this->ctx_)->const_pointer();
  }
#endif

  // Output
  Tw *y = outputs[0]->cast_data_and_get_pointer<Tw>(this->ctx_, true);
  void *m = batch_mean->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch mean
  void *v = batch_var->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch var
  // Inputs/Outputs
  void *rm = !update_inputs ? nullptr : inputs[3]
                                            ->data()
                                            ->cast(DRV_BN_T(), this->ctx_)
                                            ->pointer(); // running mean
  void *rv = !update_inputs ? nullptr : inputs[4]
                                            ->data()
                                            ->cast(DRV_BN_T(), this->ctx_)
                                            ->pointer(); // running var

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  // Get buffers.
  NdArray workspace(Shape_t{(Size_t)forward_workspace_size_});
  reserve_ = make_shared<NdArray>(Shape_t{(Size_t)reserve_size_});
  void *workspace_ptr = workspace.cast(DRV_BN_T(), this->ctx_, true)->pointer();
  void *reserve_ptr = reserve_->cast(DRV_BN_T(), this->ctx_, true)->pointer();
  // Execute forward.
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
      this->cudnn_handle_, this->mode_, this->ops_, &a, &b, input_desc_.desc,
      x,                    /* x */
      z_desc_.desc, z,      /* z */
      output_desc_.desc, y, /* y */
      this->bn_scale_bias_mean_var_desc_.desc, gamma, beta,
      1 - this->decay_rate_, rm, rv, eps, m, v,
      this->act_desc_.desc,    /* activation descriptor */
      workspace_ptr,           /* workspace pointer */
      forward_workspace_size_, /* workspace size */
      reserve_ptr,             /* reserve space pointer */
      reserve_size_            /* reserve space size */
      ));
}

template <class T>
void FusedBatchNormalizationCudaCudnn<T>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  fused_batch_norm_forward(inputs, outputs, true /* update_inputs */);
}

template <class T>
void FusedBatchNormalizationCudaCudnn<T>::recompute_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &need_recompute) {
  fused_batch_norm_forward(inputs, outputs, false /* update_inputs */);
}

template <class T>
void FusedBatchNormalizationCudaCudnn<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(this->batch_stat_, error_code::runtime,
             "If batch_stat is false, this function should not be called.");
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2] ||
        (inputs.size() == 6 && propagate_down[5]))) {
    return;
  }
  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  // Common inputs wrt. gradient.
  const Tw *dy = outputs[0]->get_grad_pointer<Tw>(this->ctx_);
  const Tw *y = outputs[0]->get_data_pointer<Tw>(this->ctx_);
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
  NdArray prop_down_workspace;
  if (prop_down_workspace_size) {
    prop_down_workspace.reshape({static_cast<Size_t>(prop_down_workspace_size)},
                                true);
    prop_down_buf = prop_down_workspace.cast(dtypes::BYTE, this->ctx_, true)
                        ->pointer<void>();
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

  // Get buffers.
  NdArray workspace(Shape_t{(Size_t)backward_workspace_size_});
  NBLA_CHECK(reserve_, error_code::value, "Forward is not called.");
  void *workspace_ptr = workspace.cast(DRV_BN_T(), this->ctx_, true)->pointer();
  void *reserve_ptr =
      reserve_->cast(DRV_BN_T(), this->ctx_, false /* rw access */)->pointer();

  void *dz =
      (inputs.size() == 6 && propagate_down[5])
          ? inputs[5]->cast_grad_and_get_pointer<Tw>(this->ctx_, !accum[5])
          : nullptr;

  // Just garbage
  NdArray prop_down_dz_buf(inputs[0]->shape());
  if (inputs.size() == 6 && !propagate_down[5]) {
    dz = prop_down_dz_buf.cast(DRV_BN_T(), this->ctx_, true)->pointer();
  }

#if defined(WORKAROUND_FOR_BUG_OPS_BN_ACTIVATION)
  if (dz == nullptr) {
    dz = prop_down_dz_buf.cast(DRV_BN_T(), this->ctx_, true)->pointer();
  }
#endif

  // Execute backward.
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
      this->cudnn_handle_, this->mode_, this->ops_, &a_data, &b_data, &a_param,
      &b_param, input_desc_.desc, x, /* x */
      output_desc_.desc, y,          /* y */
      output_desc_.desc, dy,         /* dy */
      z_desc_.desc, dz,              /* dz */
      input_desc_.desc, dx,          /* dx */
      this->bn_scale_bias_mean_var_desc_.desc, gamma, beta, dg, db, eps, m, v,
      this->act_desc_.desc,     /* activation descriptor */
      workspace_ptr,            /* workspace pointer */
      backward_workspace_size_, /* workspace size */
      reserve_ptr,              /* reserve space pointer */
      reserve_size_             /* reserve space size */
      ));
  // Clear reserved buffer for backward
  reserve_ = nullptr;
}
#endif
} // namespace nbla
