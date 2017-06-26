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
#include <nbla/cuda/solver/adamax.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_adamax_update(const int num, T *theta, T *m, T *u,
                                     const T *g, const float alpha_t,
                                     const float beta1, const float beta2,
                                     const float eps) {
  NBLA_CUDA_KERNEL_LOOP(s, num) {
    // Updating running mean and var.
    m[s] = beta1 * m[s] + (1 - beta1) * g[s];
    u[s] = max(beta2 * u[s], abs(g[s]));
    // Update parameters.
    theta[s] = theta[s] - alpha_t * m[s] / (u[s] + eps);
  }
}

template <typename T>
__global__ void kernel_weight_decay(const int num, T *grad, const T *data,
                                    const float decay_rate) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] += decay_rate * data[idx]; }
}

template <typename T>
void AdamaxCuda<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = this->state_.at(key);
  int &t = state.t;
  VariablePtr s1 = state.mean;
  VariablePtr s2 = state.u;
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *u = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<int>::max());
  const T bias_correction = 1 / (1 - std::pow(this->beta1_, t));
  const T alpha_t = this->alpha_ * bias_correction;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_adamax_update, size, theta, m, u, g,
                                 alpha_t, this->beta1_, this->beta2_,
                                 this->eps_);
}
NBLA_DEF_WEIGHT_DECAY(AdamaxCuda, weight_decay_cuda);

// Template instantiation
template class AdamaxCuda<float>;
}
