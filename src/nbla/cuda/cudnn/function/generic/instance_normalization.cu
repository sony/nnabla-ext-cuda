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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/instance_normalization.hpp>
#include <nbla/variable.hpp>

namespace nbla {

inline void dump_buffer(const float *ptr, const int size, string message) {
  std::cout << message << ": ";
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << ", ";
  }
  std::cout << std::endl;
}

inline void dump_data_buffer(Variable *var, const int size, string message) {
  cudaDeviceSynchronize();
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  const float *ptr = var->get_data_pointer<float>(cpu_ctx);
  dump_buffer(ptr, size, message);
}

inline void dump_grad_buffer(Variable *var, const int size, string message) {
  cudaDeviceSynchronize();
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  const float *ptr = var->get_grad_pointer<float>(cpu_ctx);
  dump_buffer(ptr, size, message);
}

#define DRV_BN_T() get_dtype_by_cudnn_data_type(derived_bn_dtype_)

template <typename T>
void InstanceNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                   const Variables &outputs) {
  InstanceNormalizationCuda<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (outputs.size() == 3) {
    // [WORKAROUND]
    // To use saved mean and variance and to propagate mean and variance
    // gradient are not supported with cuDNN.
    // Because cuDNN's backward interface is different from NNabla's one.
    // So Fall back to CUDA implementation if outputs.size() == 3
    // TODO: Change saved variance to inverse variance like cuDNN
    this->fall_back_func_ = make_shared<InstanceNormalizationCuda<T>>(
        this->ctx_, this->channel_axis_, this->batch_axis_, this->eps_,
        this->no_scale_, this->no_bias_);
    this->fall_back_func_->setup(inputs, outputs);
    return;
  }

  cudnn_handle_ = SingletonManager::get<CudnnHandleManager>()->handle(device_);

  channel_last_ =
      this->channel_axis_ == inputs[0]->ndim() - 1 && inputs[0]->ndim() != 2;

  if (channel_last_) {
    const int c = this->channel_axis_;
    const auto ndim = inputs[0]->ndim();

    // Pre-transpose: [b, h, w, c] -> [b, c, h, w]
    vector<int> pre_transpose_shape;
    for (int i = 0; i < this->batch_axis_.size(); i++) {
      pre_transpose_shape.push_back(i);
    }
    pre_transpose_shape.push_back(c);
    for (int i = this->batch_axis_.size(); i < ndim - 1; i++) {
      pre_transpose_shape.push_back(i);
    }
    pre_transpose_ = create_Transpose(this->ctx_, pre_transpose_shape);
    pre_transpose_->setup({inputs[0]}, {&pre_adaptor_});

    // Post-transpose: [b, c, h, w] -> [b, h, w, c]
    vector<int> post_transpose_shape;
    for (int i = 0; i < this->batch_axis_.size(); i++) {
      post_transpose_shape.push_back(i);
    }
    for (int i = this->batch_axis_.size(); i < ndim - 1; i++) {
      post_transpose_shape.push_back(i + 1);
    }
    post_transpose_shape.push_back(this->batch_axis_.size());
    post_transpose_ = create_Transpose(this->ctx_, post_transpose_shape);
    post_adaptor_.reshape(pre_adaptor_.shape(), true);
    post_transpose_->setup({&post_adaptor_}, {outputs[0]});

    reduction_size_ = pre_adaptor_.size(this->batch_axis_.size() + 1);
    outer_size_ = pre_adaptor_.size() / reduction_size_;
  } else {
    reduction_size_ = inputs[0]->size(this->channel_axis_ + 1);
    outer_size_ = inputs[0]->size() / reduction_size_;
  }

  // [N, C, H, W] -> [1, N*C, H*W, 1]
  int N = 1;
  int C = outer_size_;
  int H = reduction_size_;
  int W = 1;

  mode_ = CUDNN_BATCHNORM_SPATIAL;
  NBLA_CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(input_desc_.desc, CUDNN_TENSOR_NCHW,
                                 cudnn_data_type<T>::type(), N, C, H, W));
  NBLA_CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(output_desc_.desc, CUDNN_TENSOR_NCHW,
                                 cudnn_data_type<T>::type(), N, C, H, W));

  // Get BN data type.
  NBLA_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
      bn_scale_bias_mean_var_desc_.desc, input_desc_.desc, mode_));
  int n, c, h, w, sn, sc, sh, sw; // garbage
  NBLA_CUDNN_CHECK(cudnnGetTensor4dDescriptor(bn_scale_bias_mean_var_desc_.desc,
                                              &derived_bn_dtype_, &n, &c, &h,
                                              &w, &sn, &sc, &sh, &sw));

  b_idx_ = 1;
  g_idx_ = this->no_bias_ ? 1 : 2;

  mean_.reshape({outer_size_}, true);
  var_.reshape({outer_size_}, true);
  beta_dummy_.reshape({outer_size_}, true);
  gamma_dummy_.reshape({outer_size_}, true);
}

template <typename T>
void InstanceNormalizationCudaCudnn<T>::forward_impl(const Variables &inputs,
                                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  if (channel_last_) {
    pre_transpose_->forward({inputs[0]}, {&pre_adaptor_});

    auto in_cf_in = inputs;
    in_cf_in[0] = &pre_adaptor_;
    auto in_cf_out = outputs;
    in_cf_out[0] = &post_adaptor_;

    forward_channel_first(in_cf_in, in_cf_out);

    post_transpose_->forward({&post_adaptor_}, {outputs[0]});
  } else {
    forward_channel_first(inputs, outputs);
  }
}

template <typename T>
void InstanceNormalizationCudaCudnn<T>::forward_channel_first(
    const Variables &inputs, const Variables &outputs) {
  InstanceNormalizationCuda<T>::forward_channel_first(inputs, outputs);
  return;

  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;

  // Inputs
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);

  // dummy beta, gamma variables
  const auto param_shape = this->mean_.shape();
  if (this->no_bias_) {
    beta_dummy_.reshape(param_shape, true);
    beta_dummy_.data()->zero();
  }
  if (this->no_scale_) {
    gamma_dummy_.reshape(param_shape, true);
    gamma_dummy_.data()->fill(1.);
  }

  const void *beta =
      this->no_bias_
          ? beta_dummy_.data()->get(DRV_BN_T(), this->ctx_)->const_pointer()
          : inputs[this->b_idx_]
                ->data()
                ->get(DRV_BN_T(), this->ctx_)
                ->const_pointer();

  const void *gamma =
      this->no_scale_
          ? gamma_dummy_.data()->get(DRV_BN_T(), this->ctx_)->const_pointer()
          : inputs[this->g_idx_]
                ->data()
                ->get(DRV_BN_T(), this->ctx_)
                ->const_pointer();

  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  void *m = batch_mean->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch mean
  void *v = batch_var->data()
                ->cast(DRV_BN_T(), this->ctx_, true)
                ->pointer(); // batch var
  // Running mean and variance.
  void *rm = nullptr;
  void *rv = nullptr;

  auto a = get_cudnn_scalar_arg<T>(1);
  auto b = get_cudnn_scalar_arg<T>(0);
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      cudnn_handle_, mode_, &a, &b, input_desc_.desc, x, output_desc_.desc, y,
      bn_scale_bias_mean_var_desc_.desc, gamma, beta, 0, rm, rv, eps, m, v));
}

template <typename T>
void InstanceNormalizationCudaCudnn<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);

  if (channel_last_) {
    auto in_cf_in = inputs;
    in_cf_in[0] = &pre_adaptor_;
    auto in_cf_out = outputs;
    in_cf_out[0] = &post_adaptor_;

    post_transpose_->backward({&post_adaptor_}, {outputs[0]}, {true}, {false});

    auto in_cf_accum = accum;
    in_cf_accum[0] = false;
    backward_channel_first(in_cf_in, in_cf_out, propagate_down, in_cf_accum);

    pre_transpose_->backward({inputs[0]}, {&pre_adaptor_}, {propagate_down[0]},
                             {accum[0]});
  } else {
    backward_channel_first(inputs, outputs, propagate_down, accum);
  }
}

template <typename T>
void InstanceNormalizationCudaCudnn<T>::backward_channel_first(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  InstanceNormalizationCuda<T>::backward_channel_first(inputs, outputs,
                                                       propagate_down, accum);
  return;

  const bool pd_beta = !this->no_bias_ && propagate_down[this->b_idx_];
  const bool pd_gamma = !this->no_scale_ && propagate_down[this->g_idx_];

  const bool accum_beta = !this->no_bias_ && accum[this->b_idx_];
  const bool accum_gamma = !this->no_scale_ && accum[this->g_idx_];

  // Check whether it outputs batch mean/var.
  Variable *batch_mean = &this->mean_;
  Variable *batch_var = &this->var_;
  // Common inputs wrt. gradient.
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const void *m =
      batch_mean->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const void *v =
      batch_var->data()->get(DRV_BN_T(), this->ctx_)->const_pointer();
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);

  auto a_data = get_cudnn_scalar_arg<T>(propagate_down[0] ? 1 : 0);
  auto b_data = get_cudnn_scalar_arg<T>(accum[0] && propagate_down[0] ? 1 : 0);
  auto a_param = get_cudnn_scalar_arg<T>((pd_beta || pd_gamma) ? 1 : 0);
  auto b_param = get_cudnn_scalar_arg<T>(
      ((pd_beta && accum_beta) || (pd_gamma && accum_gamma)) ? 1 : 0);

  size_t prop_down_workspace_size = 0;
  if (!propagate_down[0]) {
    prop_down_workspace_size = std::max(
        prop_down_workspace_size, inputs[0]->size() * sizeof_dtype(DRV_BN_T()));
  }
  if (!pd_beta || !pd_gamma) {
    prop_down_workspace_size = std::max(prop_down_workspace_size,
                                        outer_size_ * sizeof_dtype(DRV_BN_T()));
  }
  void *prop_down_buf = nullptr;
  NdArray prop_down_workspace;
  if (prop_down_workspace_size) {
    prop_down_workspace.reshape({static_cast<Size_t>(prop_down_workspace_size)},
                                true);
    prop_down_buf = prop_down_workspace.cast(dtypes::BYTE, this->ctx_, true)
                        ->pointer<void>();
  }

  Tc *dx = propagate_down[0]
               ? inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0])
               : (Tc *)prop_down_buf;

  // dummy beta, gamma variables
  // Variable beta_dummy, gamma_dummy;
  // const auto param_shape = this->mean_.shape();
  // if (this->no_bias_) {
  //   beta_dummy.reshape(param_shape, true);
  //   beta_dummy.data()->zero();
  // }
  if (this->no_scale_) {
    gamma_dummy_.data()->fill(1.);
  }

  // const void *beta =
  //     this->no_bias_
  //         ? beta_dummy.data()->get(DRV_BN_T(), this->ctx_)->const_pointer()
  //         : inputs[this->b_idx_]
  //               ->data()
  //               ->get(DRV_BN_T(), this->ctx_)
  //               ->const_pointer();

  const void *gamma =
      this->no_scale_
          ? gamma_dummy_.data()->get(DRV_BN_T(), this->ctx_)->const_pointer()
          : inputs[this->g_idx_]
                ->data()
                ->get(DRV_BN_T(), this->ctx_)
                ->const_pointer();

  // Specify write only flag to prevent unnecessary memset.
  const bool param_diff_write = b_param == 0;
  void *db = pd_beta
                 ? inputs[this->b_idx_]
                       ->grad()
                       ->cast(DRV_BN_T(), this->ctx_, param_diff_write)
                       ->pointer()
                 : prop_down_buf;
  void *dg = pd_gamma
                 ? inputs[this->g_idx_]
                       ->grad()
                       ->cast(DRV_BN_T(), this->ctx_, param_diff_write)
                       ->pointer()
                 : prop_down_buf;
  double eps = std::max((double)this->eps_, CUDNN_BN_MIN_EPSILON);
  NBLA_CUDNN_CHECK(cudnnBatchNormalizationBackward(
      cudnn_handle_, mode_, &a_data, &b_data, &a_param, &b_param,
      input_desc_.desc, x, output_desc_.desc, dy, input_desc_.desc, dx,
      bn_scale_bias_mean_var_desc_.desc, gamma, dg, db, eps, m, v));
}
}
