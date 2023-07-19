// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/solver/lars.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>

#include "./clip_grad.cuh"
#include "./mixed_precision_training.cuh"
#include "./weight_decay.cuh"

namespace nbla {

constexpr int blocks = 1024; /* max blocks */

template <typename T>
__global__ void kernel_reduce_pow2_per_block(const int N, const T *x1, T *buff1,
                                             const T *x2, T *buff2) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data1 = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data1 += (AccT)x1[i] * (AccT)x1[i]; }
  thread_data1 = blockReduceSum(thread_data1);
  if (threadIdx.x == 0) {
    buff1[blockIdx.x] = thread_data1;
  }

  AccT thread_data2 = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data2 += (AccT)x2[i] * (AccT)x2[i]; }
  thread_data2 = blockReduceSum(thread_data2);
  if (threadIdx.x == 0) {
    buff2[blockIdx.x] = thread_data2;
  }
}
template <typename T>
__global__ void kernel_reduce_per_block(const int N, const T *x1, T *buff1,
                                        const T *x2, T *buff2) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data1 = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data1 += (AccT)x1[i]; }
  thread_data1 = blockReduceSum(thread_data1);
  if (threadIdx.x == 0) {
    buff1[blockIdx.x] = thread_data1;
  }

  AccT thread_data2 = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data2 += (AccT)x2[i]; }
  thread_data2 = blockReduceSum(thread_data2);
  if (threadIdx.x == 0) {
    buff2[blockIdx.x] = thread_data2;
  }
}
template <typename T>
void sq_sum(cudaStream_t stream, const int num, const T *data, T *buff1,
            T *sq_data, const T *grad, T *buff2, T *sq_grad) {
  if (num >= 1024) {
    int blocks = min(NBLA_CUDA_GET_BLOCKS(num), /*max blocks*/ 1024);
    kernel_reduce_pow2_per_block<<<blocks, NBLA_CUDA_NUM_THREADS, 0, stream>>>(
        num, data, buff1, grad, buff2);
    kernel_reduce_per_block<<<1, 1024, 0, stream>>>(blocks, buff1, sq_data,
                                                    buff2, sq_grad);
  } else {
    kernel_reduce_pow2_per_block<<<1, 1024, 0, stream>>>(num, data, sq_data,
                                                         grad, sq_grad);
  }
}

template <typename T>
__global__ void kernel_lars_update(const int num, T *data, const T *grad, T *v,
                                   T *d_sq, T *g_sq, float lr, float momentum,
                                   float decay_rate, float coefficient,
                                   float eps) {
  /* Calculate L2 norm */
  auto g_norm = std::sqrt(*g_sq);
  auto d_norm = std::sqrt(*d_sq);

  /* Calculate local learning rate */
  auto x = g_norm + decay_rate * d_norm;
  if (x < eps) {
    x += eps;
  }
  float local_lr = 1;
  if (d_norm >= eps) {
    local_lr = coefficient * d_norm / x;
  }

  // Update weight and momentum
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    v[idx] = momentum * v[idx] +
             lr * local_lr * (grad[idx] + decay_rate * data[idx]);
    data[idx] -= v[idx];
  }
}

template <typename T>
void LarsCuda<T>::update_impl(const string &key, VariablePtr param) {
  cuda_set_device(std::stoi(this->ctx_.device_id));

  typedef typename CudaType<T>::type Tc;
  dtypes dtype = get_dtype<Tc>();
  auto g_sq_arr = make_shared<NdArray>(Shape_t{1});
  auto d_sq_arr = make_shared<NdArray>(Shape_t{1});
  Tc *g_sq = g_sq_arr->cast(dtype, this->ctx_)->pointer<Tc>();
  Tc *d_sq = d_sq_arr->cast(dtype, this->ctx_)->pointer<Tc>();

  NdArray d_buff_arr(Shape_t{blocks});
  Tc *d_buff = d_buff_arr.cast(dtype, this->ctx_, true)->pointer<Tc>();
  NdArray g_buff_arr(Shape_t{blocks});
  Tc *g_buff = g_buff_arr.cast(dtype, this->ctx_, true)->pointer<Tc>();

  Size_t size = param->size();
  VariablePtr v_var = this->states_.at(key).pstate["v"];
  Tc *v = v_var->cast_data_and_get_pointer<Tc>(this->ctx_);
  Tc *data = param->cast_data_and_get_pointer<Tc>(this->ctx_);
  const Tc *grad = param->get_grad_pointer<Tc>(this->ctx_);

  /* calculate squared sum */
  sq_sum(nullptr, size, data, d_buff, d_sq, grad, g_buff, g_sq);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_lars_update, size, data, grad, v, d_sq,
                                 g_sq, this->lr_, this->momentum_,
                                 this->weight_decay_rate_, this->coefficient_,
                                 this->eps_);

  auto &t = this->states_.at(key).t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(LarsCuda, weight_decay_cuda);
NBLA_DEF_CLIP_GRAD_BY_NORM(LarsCuda, clip_grad_by_norm_cuda);
NBLA_DEF_CHECK_INF_GRAD(LarsCuda, check_inf_grad_cuda);
NBLA_DEF_CHECK_NAN_GRAD(LarsCuda, check_nan_grad_cuda);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(LarsCuda, check_inf_or_nan_grad_cuda);
NBLA_DEF_SCALE_GRAD(LarsCuda, scale_grad_impl_cuda);
} // namespace nbla
