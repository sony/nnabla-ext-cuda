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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/mean_subtraction.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void MeanSubtractionCuda<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->update_running_mean_) { // Training mode.
    forward_impl_batch(inputs, outputs);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <typename T>
void MeanSubtractionCuda<T>::recompute_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &need_recompute) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  forward_impl_global(inputs, outputs);
}

template <typename T>
__global__ void kernel_mean_subtraction_inc_t(T *t, const int max) {
  if (t[0] < max) {
    t[0] = t[0] + 1;
  }
}

template <typename T>
__global__ void kernel_mean_subtraction_forward_batch(const int size1_,
                                                      const int size0_,
                                                      const T *x, T *m, T *rm,
                                                      T *y, const int *t) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1_) {
    T coef = 1.0 / ((*t) + 1);

    // Batch mean
    T mean = 0;
    for (int i0 = 0; i0 < size0_; ++i0) {
      mean += x[i1 + i0 * size1_];
    }
    m[i1] = mean / size0_;

    // Moving mean
    rm[i1] = rm[i1] + (m[i1] - rm[i1]) * coef;

    // Output
    for (int i0 = 0; i0 < size0_; ++i0) {
      y[i1 + i0 * size1_] = x[i1 + i0 * size1_] - rm[i1];
    }
  }
}

template <class T>
void MeanSubtractionCuda<T>::forward_impl_batch(const Variables &inputs,
                                                const Variables &outputs) {
  // Inputs
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Variable *batch_mean = &this->mean_;
  Tc *m =
      batch_mean->cast_data_and_get_pointer<Tc>(this->ctx_, true); // batch mean

  // Inputs/Outputs
  Tc *rm = inputs[1]->cast_data_and_get_pointer<Tc>(this->ctx_); // running mean
  int *t =
      inputs[2]->cast_data_and_get_pointer<int>(this->ctx_); // running count

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_mean_subtraction_forward_batch,
                                 this->size1_, this->size0_, x, m, rm, y, t);

  kernel_mean_subtraction_inc_t<<<1, 1>>>(t, std::numeric_limits<int>::max());
}

template <typename T>
__global__ void
kernel_mean_subtraction_forward_global(const int size1_, const int size0_,
                                       const T *x, const T *rm, T *y) {
  NBLA_CUDA_KERNEL_LOOP(i1, size1_) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      y[i1 + i0 * size1_] = x[i1 + i0 * size1_] - rm[i1];
    }
  }
}

template <class T>
void MeanSubtractionCuda<T>::forward_impl_global(const Variables &inputs,
                                                 const Variables &outputs) {
  // Inputs
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *rm = inputs[1]->get_data_pointer<Tc>(this->ctx_); // running mean

  // Output
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_mean_subtraction_forward_global,
                                 this->size1_, this->size0_, x, rm, y);
}

template <typename T>
void MeanSubtractionCuda<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->update_running_mean_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    backward_impl_global(inputs, outputs, propagate_down, accum);
  }
}

template <typename T, bool accum>
__global__ void
kernel_mean_subtraction_backward_batch(const int num, T *dx, const T *dy,
                                       const int *t, const int size0_) {
  const T factor = (T)1.0 / ((*t) * size0_);
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : (T)0) + dy[idx] * (1 - factor);
  }
}

template <class T>
void MeanSubtractionCuda<T>::backward_impl_batch(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const int *t = inputs[2]->get_data_pointer<int>(this->ctx_);
  size_t size = inputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_batch<Tc, true>), size, dx, dy, t,
        this->size0_);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_batch<Tc, false>), size, dx, dy, t,
        this->size0_);
  }
}

template <typename T, bool accum>
__global__ void kernel_mean_subtraction_backward_global(const int num, T *dx,
                                                        const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    dx[idx] = (accum ? dx[idx] : (T)0) + dy[idx];
  }
}

template <class T>
void MeanSubtractionCuda<T>::backward_impl_global(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  size_t size = inputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_global<Tc, true>), size, dx, dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_mean_subtraction_backward_global<Tc, false>), size, dx, dy);
  }
}
}
