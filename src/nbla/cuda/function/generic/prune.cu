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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/function/prune.hpp>
#include <nbla/variable.hpp>

#include <cstdlib>
#include <memory>

#include <stdint.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace nbla {

template <typename T>
__global__ void kernel_abs_copy(const int num, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = abs(x[idx]); }
}

template <typename T, bool is_rate_one>
__global__ void kernel_thresh_forward(const int num, T *y, const T *x, T *buff,
                                      const int th_index) {
  // threshold value
  T thresh_val = buff[th_index];
  if (is_rate_one)
    thresh_val += 1.0;

  // pruning
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    y[idx] = (abs(x[idx]) < thresh_val) ? (T)0 : x[idx];
  }
}

template <typename T, bool accum = true>
__global__ void kernel_thresh_backward(const int num, T *dx, const T *x,
                                       const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : (T)0) + dy[idx];
  }
}

template <typename T>
void PruneCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Prune<T>::setup_impl(inputs, outputs);
  cuda_set_device(std::stoi(this->ctx_.device_id));
}

template <typename T>
void PruneCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {

  cuda_set_device(std::stoi(this->ctx_.device_id));
  // fetch input and output
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Size_t size = inputs[0]->size();
  dtypes dtype = get_dtype<Tc>();
  ArrayPtr array = make_shared<CudaCachedArray>(size, dtype, this->ctx_);
  auto buffer_ = array->pointer<T>();

  // copy
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_abs_copy, size, buffer_, x);

  // perform sort
  auto buf_vec = thrust::device_vector<Tc>(buffer_, buffer_ + size);
  thrust::sort(buf_vec.begin(), buf_vec.end());

  // TODO: without this logic, the result is inconsistent.
  thrust::copy(buf_vec.begin(), buf_vec.end(), buffer_);

  // prune
  if (this->rate_ == 1.0) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_thresh_forward<Tc, true>), size, y,
                                   x, buffer_, this->thresh_idx_);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_thresh_forward<Tc, false>), size, y,
                                   x, buffer_, this->thresh_idx_);
  }
}

template <typename T>
void PruneCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Size_t size = inputs[0]->size();

  if (propagate_down[0]) {
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_thresh_backward<Tc, true>), size,
                                     dx, x, dy);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_thresh_backward<Tc, false>), size,
                                     dx, x, dy);
    }
  }
}
}
