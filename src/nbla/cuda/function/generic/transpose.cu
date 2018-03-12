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

// transpose.cpp

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/transpose.hpp>

namespace nbla {

template <typename T>
__global__ void
kernel_transpose_forward(const int num, const int ndim, const int64_t *axes,
                         const int64_t *x_strides, const int64_t *y_strides,
                         const int64_t *y_shape, const T *x, T *y) {

  NBLA_CUDA_KERNEL_LOOP(o, num) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / y_strides[d]) % y_shape[d];
      i += k * x_strides[axes[d]];
    }
    y[o] = x[i];
  }
}

template <typename T, bool accum>
__global__ void
kernel_transpose_backward(const int num, const int ndim, const int64_t *axes,
                          const int64_t *x_strides, const int64_t *y_strides,
                          const int64_t *y_shape, const T *dy, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(o, num) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / y_strides[d]) % y_shape[d];
      i += k * x_strides[axes[d]];
    }
    dx[i] = (accum ? dx[i] : (T)0) + dy[o];
  }
}

template <typename T>
void TransposeCuda<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  Transpose<T>::setup_impl(inputs, outputs);
}

template <class T>
void TransposeCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);

  // To avoid compiler error : type name is not allowed.
  // The following statement causes a compiler error.
  // this->v_axes_.get_data_pointer<int64_t>(this->ctx_)
  auto get_ = [this](Variable &var) {
    return var.get_data_pointer<int64_t>(this->ctx_);
  };
  const int64_t *axes = get_(this->v_axes_);
  const int64_t *x_strides = get_(this->v_x_strides_);
  const int64_t *y_strides = get_(this->v_y_strides_);
  const int64_t *y_shape = get_(this->v_y_shape_);
  const int ndim = inputs[0]->ndim();
  const int size = outputs[0]->size();

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_transpose_forward, size, ndim, axes,
                                 x_strides, y_strides, y_shape, x, y);
}

template <class T>
void TransposeCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_);

  // To avoid compiler error : type name is not allowed.
  // The following statement causes a compiler error.
  // this->v_axes_.get_data_pointer<int64_t>(this->ctx_)
  auto get_ = [this](Variable &var) {
    return var.get_data_pointer<int64_t>(this->ctx_);
  };
  const int64_t *axes = get_(this->v_axes_);
  const int64_t *x_strides = get_(this->v_x_strides_);
  const int64_t *y_strides = get_(this->v_y_strides_);
  const int64_t *y_shape = get_(this->v_y_shape_);
  const int ndim = inputs[0]->ndim();
  const int size = outputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_transpose_backward<Tc, true>), size,
                                   ndim, axes, x_strides, y_strides, y_shape,
                                   dy, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_transpose_backward<Tc, false>), size,
                                   ndim, axes, x_strides, y_strides, y_shape,
                                   dy, dx);
  }
}
}
