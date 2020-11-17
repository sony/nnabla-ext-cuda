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
#include <nbla/cuda/function/scatter_add.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

namespace scatter_add_cuda {
template <typename T>
__global__ void forward_x0(const int x0_size, const T *x0_data, T *y_data) {
  NBLA_CUDA_KERNEL_LOOP(tid, x0_size) { y_data[tid] = x0_data[tid]; }
}

template <typename T>
__global__ void forward_x1(const int indices_size, const int *indices_data,
                           const int *indices_stride, const int *x0_stride,
                           const int ndim, const T *x1_data,
                           const int *x1_stride, T *y_data, const int axis) {
  NBLA_CUDA_KERNEL_LOOP(tid, indices_size) {
    auto dst_axis_index = indices_data[tid];
    int src_flat_index = 0;
    int dst_flat_index = 0;
    auto index = tid;
    for (int d = 0; d < ndim; d++) {
      auto nd_index = index / indices_stride[d];
      index -= nd_index * indices_stride[d];
      src_flat_index += nd_index * x1_stride[d];
      if (d == axis) {
        dst_flat_index += dst_axis_index * x0_stride[d];
      } else {
        dst_flat_index += nd_index * x0_stride[d];
      }
    }
    atomic_add(y_data + dst_flat_index, x1_data[src_flat_index]);
  }
}

template <typename T, bool accum = true>
__global__ void backward_x0(const int x0_size, T *x0_grad, const T *y_grad) {
  NBLA_CUDA_KERNEL_LOOP(tid, x0_size) {
    if (accum) {
      x0_grad[tid] = x0_grad[tid] + y_grad[tid];
    } else {
      x0_grad[tid] = y_grad[tid];
    }
  }
}

template <typename T, bool accum = true>
__global__ void backward_x1(const int indices_size, const int *indices_data,
                            const int *indices_stride, const int *x0_stride,
                            const int ndim, const int x1_size, T *x1_grad,
                            const int *x1_stride, const T *y_grad,
                            const int axis) {
  if (!accum) {
    NBLA_CUDA_KERNEL_LOOP(tid, x1_size) { x1_grad[tid] = 0; }
  }
  NBLA_CUDA_KERNEL_LOOP(tid, indices_size) {
    auto dst_axis_index = indices_data[tid];
    int src_flat_index = 0;
    int dst_flat_index = 0;
    auto index = tid;
    for (int d = 0; d < ndim; d++) {
      auto nd_index = index / indices_stride[d];
      index -= nd_index * indices_stride[d];
      src_flat_index += nd_index * x1_stride[d];
      if (d == axis) {
        dst_flat_index += dst_axis_index * x0_stride[d];
      } else {
        dst_flat_index += nd_index * x0_stride[d];
      }
    }
    if (accum) {
      x1_grad[src_flat_index] =
          x1_grad[src_flat_index] + y_grad[dst_flat_index];
    } else {
      x1_grad[src_flat_index] = y_grad[dst_flat_index];
    }
  }
}
}

template <typename T>
void ScatterAddCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  ScatterAdd<T>::setup_impl(inputs, outputs);
  // All the inputs must have same dimension
  Shape_t meta_shape = {2 * inputs[0]->ndim()};
  x0_meta_.reshape(meta_shape, true);
  indices_meta_.reshape(meta_shape, true);
  x1_meta_.reshape(meta_shape, true);

  Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  copy_meta(inputs[0], x0_meta_, cpu_ctx);
  copy_meta(inputs[1], indices_meta_, cpu_ctx);
  copy_meta(inputs[2], x1_meta_, cpu_ctx);
}

template <typename T>
void ScatterAddCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);

  auto x0_shape = inputs[0]->shape();
  auto indices_shape = inputs[1]->shape();
  auto x1_shape = inputs[2]->shape();

  // Inputs
  auto x0 = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto indices = inputs[1]->get_data_pointer<int>(this->ctx_);
  auto x1 = inputs[2]->get_data_pointer<Tcu>(this->ctx_);

  auto x0_shape_ptr = x0_meta_.get_data_pointer<int>(this->ctx_);
  auto x0_stride_ptr = x0_shape_ptr + inputs[0]->ndim();

  auto indices_shape_ptr = indices_meta_.get_data_pointer<int>(this->ctx_);
  auto indices_stride_ptr = indices_shape_ptr + inputs[1]->ndim();

  auto x1_shape_ptr = x1_meta_.get_data_pointer<int>(this->ctx_);
  auto x1_stride_ptr = x1_shape_ptr + inputs[2]->ndim();

  // Outputs
  auto y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((scatter_add_cuda::forward_x0<Tcu>),
                                 inputs[0]->size(), x0, y);

  auto axis = (this->axis_ < 0) ? inputs[0]->ndim() + this->axis_ : this->axis_;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((scatter_add_cuda::forward_x1<Tcu>),
                                 inputs[1]->size(), indices, indices_stride_ptr,
                                 x0_stride_ptr, inputs[0]->ndim(), x1,
                                 x1_stride_ptr, y, axis);
}

template <typename T>
void ScatterAddCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[2])) {
    return;
  }
  cuda_set_device(this->device_);

  auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  if (propagate_down[0]) {
    auto g_x0 =
        inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((scatter_add_cuda::backward_x0<Tcu, true>),
                                     inputs[0]->size(), g_x0, g_y);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (scatter_add_cuda::backward_x0<Tcu, false>), inputs[0]->size(), g_x0,
          g_y);
    }
  }
  if (propagate_down[2]) {
    auto indices = inputs[1]->get_data_pointer<int>(this->ctx_);

    auto x0_shape_ptr = x0_meta_.get_data_pointer<int>(this->ctx_);
    auto x0_stride_ptr = x0_shape_ptr + inputs[0]->ndim();

    auto indices_shape_ptr = indices_meta_.get_data_pointer<int>(this->ctx_);
    auto indices_stride_ptr = indices_shape_ptr + inputs[1]->ndim();

    auto x1_shape_ptr = x1_meta_.get_data_pointer<int>(this->ctx_);
    auto x1_stride_ptr = x1_shape_ptr + inputs[2]->ndim();

    auto axis =
        (this->axis_ < 0) ? inputs[0]->ndim() + this->axis_ : this->axis_;

    auto g_x1 =
        inputs[2]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[2]);
    if (accum[2]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (scatter_add_cuda::backward_x1<Tcu, true>), inputs[1]->size(),
          indices, indices_stride_ptr, x0_stride_ptr, inputs[0]->ndim(),
          inputs[2]->size(), g_x1, x1_stride_ptr, g_y, axis);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (scatter_add_cuda::backward_x1<Tcu, false>), inputs[1]->size(),
          indices, indices_stride_ptr, x0_stride_ptr, inputs[0]->ndim(),
          inputs[2]->size(), g_x1, x1_stride_ptr, g_y, axis);
    }
  }
}
}
