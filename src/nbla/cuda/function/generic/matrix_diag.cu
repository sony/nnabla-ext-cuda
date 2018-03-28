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
#include <nbla/cuda/function/matrix_diag.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_matrix_diag_forward(const int num, const int last_ndim,
                                           T *y, const T *x) {
  const int num_by_last_ndim = num * last_ndim;
  NBLA_CUDA_KERNEL_LOOP(idx, num_by_last_ndim) {
    int c_idx = idx % last_ndim;   // column index in a matrix
    int x_idx = idx / last_ndim;   // index of an input array
    int r_idx = x_idx % last_ndim; // row index in a matrix
    y[idx] = (r_idx == c_idx) ? x[x_idx] : (T)0;
  }
}

template <typename T, bool accum>
__global__ void kernel_matrix_diag_backward(const int num, const int last_ndim,
                                            T *dx, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    if (accum)
      dx[idx] += dy[idx * last_ndim + idx % last_ndim];
    else
      dx[idx] = dy[idx * last_ndim + idx % last_ndim];
  }
}

template <typename T>
void MatrixDiagCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_matrix_diag_forward, size,
                                 this->last_ndim_, y, x)
}

template <typename T>
void MatrixDiagCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (!propagate_down[0]) {
    return;
  }

  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Size_t size = inputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_matrix_diag_backward<Tc, true>),
                                   size, this->last_ndim_, dx, dy);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_matrix_diag_backward<Tc, false>),
                                   size, this->last_ndim_, dx, dy);
  }
}
}
