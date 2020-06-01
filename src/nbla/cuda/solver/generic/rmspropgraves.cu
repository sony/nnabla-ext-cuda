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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/solver/rmspropgraves.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void kernel_rmspropgraves_update(const int num, T *data, const T *grad,
                                            T *n, T *g, T *d, const float lr,
                                            const float decay, const float momentum,
                                            const float eps) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    n[idx] = decay * n[idx] + (1 - decay) * grad[idx] * grad[idx];
    g[idx] = decay * g[idx] + (1 - decay) * grad[idx];
    d[idx] = (momentum) * d[idx] - lr * grad[idx] / (sqrt(n[idx] - g[idx] * g[idx] + eps));
    data[idx] += d[idx];
  }
}

template <typename T>
void RMSpropgravesCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = this->states_.at(key);
  VariablePtr s1 = state.pstate["n"];
  VariablePtr s2 = state.pstate["g"];
  VariablePtr s3 = state.pstate["d"];
  T *n = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *g = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *d = s3->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_rmspropgraves_update, size, data, grad,
                                 n, g, d, this->lr_, this->decay_,
                                 this->momentum_, this->eps_);
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(RMSpropgravesCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(RMSpropgravesCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(RMSpropgravesCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(RMSpropgravesCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(RMSpropgravesCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(RMSpropgravesCuda, scale_grad_impl_cuda);
}
