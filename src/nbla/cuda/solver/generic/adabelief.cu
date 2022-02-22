// Copyright 2020,2021 Sony Corporation.
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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/solver/adabelief.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_adabelief_update(
    const int num, T *theta, T *m, T *s, T *s_max, const T *g,
    const float alpha_t, const float beta1, const float beta2, const float eps,
    const float decay_ratio, const bool amsgrad, const bool weight_decouple,
    const bool sgd_update, const float bias_correction2) {
  NBLA_CUDA_KERNEL_LOOP(i, num) {
    // Updating running mean and var.
    m[i] = beta1 * m[i] + (1 - beta1) * g[i];
    s[i] = beta2 * s[i] + (1 - beta2) * (g[i] - m[i]) * (g[i] - m[i]);
    if (weight_decouple) {
      theta[i] = theta[i] - theta[i] * decay_ratio;
    }
    if (amsgrad) {
      s_max[i] = max(s_max[i], s[i]);
      s_max[i] += eps;
    } else {
      s[i] += eps;
    }
    // Update parameters.
    if (sgd_update) {
      theta[i] = theta[i] - alpha_t * m[i];
    } else {
      auto s_t = amsgrad ? s_max[i] : s[i];
      auto denominator = std::sqrt(s_t) / bias_correction2;
      theta[i] = theta[i] - alpha_t * m[i] / (denominator + eps);
    }
  }
}

template <typename T>
void AdaBeliefCuda<T>::update_impl(const string &key, VariablePtr param) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  uint32_t &t = state.t;
  shared_ptr<Variable> s1_ = state.pstate["mean"];
  shared_ptr<Variable> s2_ = state.pstate["var"];
  T *m = s1_->cast_data_and_get_pointer<T>(this->ctx_);
  T *s = s2_->cast_data_and_get_pointer<T>(this->ctx_);
  T *s_max = nullptr;
  if (this->amsgrad_) {
    shared_ptr<Variable> s3 = state.pstate["s_max"];
    s_max = s3->cast_data_and_get_pointer<T>(this->ctx_);
  }
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  const T beta1_t = std::pow(this->beta1_, t);
  const T beta2_t = std::pow(this->beta2_, t);
  const T bias_correction1 = 1.0 - beta1_t;
  const T bias_correction2 = std::sqrt(1.0 - beta2_t);
  float r_t = 1.0;
  float rho_t = 0.0;

  if (this->rectify_) {
    auto rho_inf = 2.0 / (1.0 - this->beta2_) - 1.0;
    rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t);
    auto r_t_numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
    auto r_t_denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
    r_t = std::sqrt(r_t_numerator / r_t_denominator);
  }

  const T *g = param->get_grad_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  const bool sgd_update = (this->rectify_ && rho_t <= 4.0);
  const float alpha_t =
      sgd_update ? this->alpha_ : this->alpha_ * r_t / bias_correction1;
  const float decay_ratio =
      (this->fixed_decay_) ? this->wd_ : this->wd_ * this->alpha_;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_adabelief_update, size, theta, m, s, s_max, g, alpha_t,
      this->beta1_, this->beta2_, this->eps_, decay_ratio, this->amsgrad_,
      this->weight_decouple_, sgd_update, bias_correction2);
}
NBLA_DEF_WEIGHT_DECAY(AdaBeliefCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdaBeliefCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(AdaBeliefCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(AdaBeliefCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdaBeliefCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(AdaBeliefCuda, scale_grad_impl_cuda);
}
