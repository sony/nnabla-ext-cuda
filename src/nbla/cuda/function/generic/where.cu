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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/where.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_where_forward(const size_t size, const size_t inner_size,
                                     const T *condition, const T *x_true,
                                     const T *x_false, T *y) {
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    const int c = index / inner_size;
    y[index] = condition[c] ? x_true[index] : x_false[index];
  }
}

template <typename T>
__global__ void
kernel_where_backward(const size_t size, const size_t inner_size,
                      const T *condition, T *g_x_true, T *g_x_false,
                      const T *g_y, bool accum_true, bool accum_false) {
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    const bool cond = condition[index / inner_size];
    if (g_x_true) {
      g_x_true[index] =
          (accum_true ? g_x_true[index] : (T)0) + (cond ? g_y[index] : (T)0);
    }
    if (g_x_false) {
      g_x_false[index] =
          (accum_false ? g_x_false[index] : (T)0) + (cond ? (T)0 : g_y[index]);
    }
  }
}

template <typename T>
void WhereCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Where<T>::setup_impl(inputs, outputs);
}

template <typename T>
void WhereCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tcu *condition = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *x_true = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *x_false = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  size_t csize = inputs[0]->size();
  size_t xsize = inputs[1]->size();
  size_t inner_size = xsize / csize;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_where_forward, xsize, inner_size,
                                 condition, x_true, x_false, y);
}

template <typename T>
void WhereCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!(propagate_down[1] || propagate_down[2])) {
    return;
  }
  cuda_set_device(this->device_);
  const Tcu *condition = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  Tcu *g_x_true{nullptr};
  Tcu *g_x_false{nullptr};

  if (propagate_down[1]) {
    g_x_true = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
  }
  if (propagate_down[2]) {
    g_x_false =
        inputs[2]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[2]);
  }
  size_t csize = inputs[0]->size();
  size_t xsize = inputs[1]->size();
  size_t inner_size = xsize / csize;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_where_backward, xsize, inner_size,
                                 condition, g_x_true, g_x_false, g_y, accum[1],
                                 accum[2]);
}
}
