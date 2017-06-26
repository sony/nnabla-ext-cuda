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

// -*- coding:utf-8 -*-
/*
 * Copyright (C) 2016 Sony Corporation
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Corporation;
 * the contents of this file is not to be disclosed to third parties, copied
 * or duplicated in any form, in whole or in part, without the prior written
 * permission of Sony Corporation.
 */

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/prelu.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/singleton_manager.hpp>

namespace nbla {

template <typename T>
__global__ void forward_prelu_kernel(const int size, const T *x, const T *w,
                                     T *y) {
  NBLA_CUDA_KERNEL_LOOP(s, size) { y[s] = (x[s] >= 0) ? x[s] : x[s] * (*w); }
}

template <typename T>
__global__ void forward_prelu_kernel_c(const int size, const int base_shape,
                                       const int base_stride, const T *x,
                                       const T *w, T *y) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    const int iw = int(s / base_stride) % base_shape;
    y[s] = (x[s] >= 0) ? x[s] : x[s] * w[iw];
  }
}

template <typename T, bool accum = true>
__global__ void backward_prelu_kernel_input(const int size, const T *dy,
                                            const T *x, const T *w, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    dx[s] = (accum ? dx[s] : 0) + ((x[s] >= 0) ? dy[s] : dy[s] * (*w));
  }
}

template <typename T, bool accum = true>
__global__ void
backward_prelu_kernel_input_c(const int size, const int base_shape,
                              const int base_stride, const T *dy, const T *x,
                              const T *w, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(s, size) {
    const int iw = int(s / base_stride) % base_shape;
    dx[s] = (accum ? dx[s] : 0) + ((x[s] >= 0) ? dy[s] : dy[s] * w[iw]);
  }
}

template <typename T>
__global__ void
backward_prelu_kernel_weights_temp(const int insize, const int outsize,
                                   const T *dy, const T *x, T *buff) {
  NBLA_CUDA_KERNEL_LOOP(s, insize) {
    buff[s] = dy[s] * x[s] * (x[s] < 0);
    for (int i = 1; i < outsize; ++i) {
      buff[s] +=
          dy[s + i * insize] * x[s + i * insize] * (x[s + i * insize] < 0);
    }
  }
}

template <typename T, bool accum>
__global__ void kernel_reduce_per_block(const int N, const T *x, T *buff) {
  T thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += x[i]; }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    buff[blockIdx.x] = (accum ? buff[blockIdx.x] : 0) + thread_data;
  }
}

template <typename T>
void PReLUCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  PReLU<T>::setup_impl(inputs, outputs);
}

template <typename T>
void PReLUCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const int size = inputs[0]->size();
  if (inputs[1]->size() == 1) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_prelu_kernel, size, x, w, y);
    return;
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_prelu_kernel_c, size,
                                 this->base_shape_, this->base_stride_, x, w,
                                 y);
}

template <typename T>
void PReLUCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const int size = inputs[0]->size();
  if (propagate_down[0]) {
    const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
    T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    if (inputs[1]->size() == 1) {
      if (accum[0]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((backward_prelu_kernel_input<T, true>),
                                       size, dy, x, w, dx);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((backward_prelu_kernel_input<T, false>),
                                       size, dy, x, w, dx);
      }
    } else {
      if (accum[0]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((backward_prelu_kernel_input_c<T, true>),
                                       size, this->base_shape_,
                                       this->base_stride_, dy, x, w, dx);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
            (backward_prelu_kernel_input_c<T, false>), size, this->base_shape_,
            this->base_stride_, dy, x, w, dx);
      }
    }
  }
  if (propagate_down[1]) {
    T *dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    const int insize = inputs[0]->size(this->base_axis_);
    const int outsize = size / insize;
    const int channels = inputs[1]->size();
    // Weight backward consists of two step.
    // 1) Element-wise backward operation (sample dimensions are reduced).
    // 2) Reduction to weight vector (or scalar) by using well-tuned cublas
    // function.
    shared_ptr<CudaCachedArray> arr_buff(
        new CudaCachedArray(insize, get_dtype<T>(), this->ctx_));
    T *buff = arr_buff->pointer<T>();
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_prelu_kernel_weights_temp, insize,
                                   outsize, dy, x, buff);
    if (channels == 1) {
      shared_ptr<CudaCachedArray> arr_buff2;
      T *buff2 = buff;
      int blocks = insize;
      if (insize >= 1024) {
        blocks = min(NBLA_CUDA_GET_BLOCKS(insize), /*max blocks*/ 1024);
        arr_buff2 =
            make_shared<CudaCachedArray>(blocks, get_dtype<T>(), this->ctx_);
        buff2 = arr_buff2->pointer<T>();
        kernel_reduce_per_block<T, false><<<blocks, NBLA_CUDA_NUM_THREADS>>>(
            insize, buff, buff2);
      }
      if (accum[1]) {
        kernel_reduce_per_block<T, true><<<1, 1024>>>(blocks, buff, dw);
      } else {
        kernel_reduce_per_block<T, false><<<1, 1024>>>(blocks, buff, dw);
      }
    } else {
      const int spatial_size = insize / channels;
      const T *ones =
          static_cast<const T *>(SingletonManager::get<NNabla>()->ones(
              spatial_size, get_dtype<T>(), this->ctx_));
      cuda_gemv(this->device_, dw, buff, spatial_size, channels, true, ones,
                spatial_size, (T)1, (T)(accum[1] ? 1 : 0));
    }
  }
}

// template instantiation
template class PReLUCuda<float>;
}
