// Copyright 2023 Sony Group Corporation.
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

#include <cassert>
#include <queue>

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/solver/lion.hpp>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

namespace {
template <typename T>
__forceinline__ __device__ T lerp(const T a, const T b, const float t) {
  return a + t * (b - a);
}
template <typename T> __forceinline__ __device__ int sign(const T x) {
  return (x > T(0)) - (x < T(0));
}
} // namespace

template <typename T>
__global__ void kernel_lion_update(const int num, const T *g, T *m, T *w,
                                   const float lr, const float beta1,
                                   const float beta2, const float decay_rate) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    auto u = sign(lerp(g[idx], m[idx], beta1));
    m[idx] = lerp(g[idx], m[idx], beta2);
    w[idx] -= lr * (u + decay_rate * w[idx]);
  }
}

template <typename T>
void LionCuda<T>::update_impl(const string &key, VariablePtr param) {
  typedef typename CudaType<T>::type Tc;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  dtypes dtype = get_dtype<Tc>();

  auto &t = this->states_.at(key).t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);

  Size_t size = param->size();
  VariablePtr m_var = this->states_.at(key).pstate["m"];
  const Tc *g = param->get_grad_pointer<Tc>(this->ctx_);
  Tc *w = param->cast_data_and_get_pointer<Tc>(this->ctx_);
  Tc *m = m_var->cast_data_and_get_pointer<Tc>(this->ctx_);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_lion_update, size, g, m, w, this->lr_,
                                 this->beta1_, this->beta2_,
                                 this->weight_decay_rate_);
}

NBLA_DEF_WEIGHT_DECAY(LionCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(LionCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(LionCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(LionCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(LionCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(LionCuda, scale_grad_impl_cuda);
} // namespace nbla
