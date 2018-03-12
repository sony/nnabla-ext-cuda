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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/function/mean.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>

namespace nbla {

template <typename T>
__global__ void kernel_reduce_per_block(const int N, const T *x, T *buff,
                                        T scale = 1) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += (AccT)x[i]; }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    buff[blockIdx.x] = thread_data * scale;
  }
}

template <typename T>
void MeanCuda<T>::forward_impl_reduce(const T *x_, T *y_, int outer_size,
                                      int reduction_size) {
  const Tc *x = reinterpret_cast<const Tc *>(x_);
  Tc *y = reinterpret_cast<Tc *>(y_);
  cuda_set_device(this->device_);
  if (outer_size == 1) {
    if (reduction_size >= 1024) {
      int blocks =
          min(NBLA_CUDA_GET_BLOCKS(reduction_size), /*max blocks*/ 1024);
      shared_ptr<CudaCachedArray> arr_buff =
          make_shared<CudaCachedArray>(blocks, get_dtype<Tc>(), this->ctx_);
      Tc *buff = arr_buff->pointer<Tc>();
      kernel_reduce_per_block<<<blocks, NBLA_CUDA_NUM_THREADS>>>(reduction_size,
                                                                 x, buff);
      NBLA_CUDA_KERNEL_CHECK();
      kernel_reduce_per_block<Tc><<<1, 1024>>>(blocks, buff, y,
                                               (T)(1. / reduction_size));
      NBLA_CUDA_KERNEL_CHECK();
    } else {
      kernel_reduce_per_block<Tc><<<1, 1024>>>(reduction_size, x, y,
                                               (T)(1. / reduction_size));
      NBLA_CUDA_KERNEL_CHECK();
    }
    return;
  }
  const Tc *ones =
      static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
          reduction_size, get_dtype<Tc>(), this->ctx_));
  cuda_gemv<Tc>(this->device_, y, x, reduction_size, outer_size, true, ones,
                reduction_size, 1. / reduction_size, 0);
}

template <typename T, bool accum>
__global__ void kernel_reduce_mean_backward(const int num, T *dx, const T *dy,
                                            T scale) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : (T)0) + scale * (*dy);
  }
}

template <typename T>
void MeanCuda<T>::backward_impl_reduce(const T *dy_, T *dx_, int outer_size,
                                       int reduction_size, bool accum) {
  const Tc *dy = reinterpret_cast<const Tc *>(dy_);
  Tc *dx = reinterpret_cast<Tc *>(dx_);
  cuda_set_device(this->device_);
  if (outer_size == 1) {
    if (accum) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_mean_backward<Tc, true>),
                                     reduction_size, dx, dy,
                                     (T)(1. / reduction_size));
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_mean_backward<Tc, false>),
                                     reduction_size, dx, dy,
                                     (T)(1. / reduction_size));
    }
    return;
  }
  const Tc *ones =
      static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
          reduction_size, get_dtype<Tc>(), this->ctx_));
  cuda_gemm<Tc>(this->device_, dx, true, dy, outer_size, 1, false, ones, 1,
                reduction_size, false, 1. / reduction_size, accum ? 1 : 0);
}
}
