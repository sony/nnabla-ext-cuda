// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/clip_grad_by_norm.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_clip_grad_by_norm_copy(int size, T *output,
                                              const T *input) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { output[idx] = input[idx]; }
}

template <typename T>
__global__ void kernel_clip_grad_by_norm_forward(const int num, T *y,
                                                 const T *x, const T *m,
                                                 const float clip_norm) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    y[idx] = (T)clip_norm * x[idx] / sqrtf(m[idx]);
  }
}

template <typename T>
void ClipGradByNormCuda<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Size_t size = inputs[0]->size();
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_clip_grad_by_norm_copy, size, y, x);
}

template <typename T, bool accum>
__global__ void
kernel_clip_grad_by_norm_backward_cuda(int size, float clip_norm, T *dx,
                                       const T *dy, const T *m) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    T _dx = (T)clip_norm * dy[idx] / sqrtf(m[idx]);
    accum ? dx[idx] += _dx : dx[idx] = _dx;
  }
}

template <typename T>
void ClipGradByNormCuda<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  cuda_set_device(this->device_);

  if (!propagate_down[0]) {
    return;
  }

  auto shape = inputs[0]->shape();
  Variable v0(shape);
  Variable v1(shape);
  Variable v2(shape);
  Variable v3(shape);
  auto intermediates0 = Variables{&v0};
  auto intermediates1 = Variables{&v1};
  auto intermediates2 = Variables{&v2};
  auto intermediates3 = Variables{&v3};

  Size_t size = inputs[0]->size();
  Tc *_m = intermediates0[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  const Tc *_dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_clip_grad_by_norm_copy, size, _m, _dy);

  // power grads by 2.
  this->pow_scalar_->setup(intermediates0, intermediates1);
  this->pow_scalar_->forward(intermediates0, intermediates1);

  // sum grads powered by 2.
  this->sum_->setup(intermediates1, intermediates2);
  this->sum_->forward(intermediates1, intermediates2);

  // broadcast
  this->broadcast_->setup(intermediates2, intermediates3);
  this->broadcast_->forward(intermediates2, intermediates3);

  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *m = intermediates3[0]->get_data_pointer<Tc>(this->ctx_);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_clip_grad_by_norm_backward_cuda<Tc, true>), size,
        this->clip_norm_, dx, dy, m);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_clip_grad_by_norm_backward_cuda<Tc, false>), size,
        this->clip_norm_, dx, dy, m);
  }
}
}
