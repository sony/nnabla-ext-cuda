// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/cublas.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/solver/lamb.hpp>
#include <nbla/cuda/utils/fused_reduce.cuh>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

template <typename T>
__global__ void
kernel_lamb_update_state(const int num, const T *w, const T *g, T *m, T *v,
                         T *r, float beta1, const float beta2,
                         const float decay_rate, const float eps,
                         const float corr1, const float corr2) {

  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    m[idx] = beta1 * m[idx] + (1 - beta1) * g[idx];
    v[idx] = beta2 * v[idx] + (1 - beta2) * g[idx] * g[idx];
    r[idx] = (m[idx] / corr1) / (std::sqrt(v[idx] / corr2) + eps);
    r[idx] += decay_rate * w[idx];
  }
}

template <typename T>
__global__ void kernel_lamb_update(const int num, T *w, const T *g, T *r,
                                   T *d_sq, T *g_sq, float eta, float gamma_l,
                                   float gamma_u, float eps) {
  /* Calculate L2 norm */
  auto g_norm = std::sqrt(*g_sq);
  auto d_norm = std::sqrt(*d_sq);

  if (d_norm < gamma_l) {
    d_norm = gamma_l;
  }
  if (d_norm > gamma_u) {
    d_norm = gamma_u;
  }

  auto local_lr = (g_norm < eps) ? 1.0 : d_norm / g_norm;

  NBLA_CUDA_KERNEL_LOOP(idx, num) { w[idx] -= eta * local_lr * r[idx]; }
}

template <typename T>
void LambCuda<T>::update_impl(const string &key, VariablePtr param) {
  typedef typename CudaType<T>::type Tc;
  // NOTE: This solver implementation accepts specifying a stream,
  // but there is no interface to pass a stream to the class/function so far.
  cudaStream_t stream = nullptr;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  dtypes dtype = get_dtype<Tc>();

  auto &t = this->states_.at(key).t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);

  Size_t size = param->size();
  VariablePtr m_var = this->states_.at(key).pstate["mean"];
  VariablePtr v_var = this->states_.at(key).pstate["var"];
  auto r_arr = make_shared<NdArray>(param->shape());
  Tc *w = param->cast_data_and_get_pointer<Tc>(this->ctx_);
  const Tc *g = param->cast_grad_and_get_pointer<Tc>(this->ctx_);
  Tc *m = m_var->cast_data_and_get_pointer<Tc>(this->ctx_);
  Tc *v = v_var->cast_data_and_get_pointer<Tc>(this->ctx_);
  T *r = r_arr->cast(dtype, this->ctx_)->template pointer<T>();

  /* Buffer */
  auto sq_arr = make_shared<NdArray>(Shape_t{2});
  Tc *sq_ptr = sq_arr->cast(dtype, this->ctx_)->pointer<Tc>();
  Tc *d_sq = sq_ptr;
  Tc *g_sq = sq_ptr + 1;

  int blocks = /*max blocks*/ 1024;
  auto arr_buff1 = make_shared<NdArray>(Shape_t{blocks});
  Tc *buff1 = arr_buff1->cast(dtype, this->ctx_, true)->template pointer<Tc>();
  auto arr_buff2 = make_shared<NdArray>(Shape_t{blocks});
  Tc *buff2 = arr_buff2->cast(dtype, this->ctx_, true)->template pointer<Tc>();

  /* update state */
  const float corr1 = this->bias_correction_ ? 1 - pow(this->beta1_, t) : 1.0f;
  const float corr2 = this->bias_correction_ ? 1 - pow(this->beta2_, t) : 1.0f;
  NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(
      kernel_lamb_update_state, stream, size, w, g, m, v, r, this->beta1_,
      this->beta2_, this->weight_decay_rate_, this->eps_, corr1, corr2);

  /* calculate squared sum */
  fused_reduce<Square<float>, Tc, Tc>(stream, size,
                                      ReduceTarget<T>{w, buff1, d_sq},
                                      ReduceTarget<T>{r, buff2, g_sq});

  /* update weight */
  NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_lamb_update, stream, size, w, g, r,
                                    d_sq, g_sq, this->eta_, this->gamma_l_,
                                    this->gamma_u_, this->eps_);
}

NBLA_DEF_WEIGHT_DECAY(LambCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(LambCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(LambCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(LambCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(LambCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(LambCuda, scale_grad_impl_cuda);
} // namespace nbla
