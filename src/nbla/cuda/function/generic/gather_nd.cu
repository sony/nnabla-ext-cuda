// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/gather_nd.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

namespace gather_nd_cuda {

template <typename T>
__global__ void forward(const int y_size, T *y_data, const int x_size,
                        const T *x_data, const int *x_shape,
                        const int *x_stride, const int *idx_data,
                        const int idx_rows, const int idx_cols) {
  NBLA_CUDA_KERNEL_LOOP(tid, y_size) {
    auto slice_length = y_size / idx_cols;
    auto index_column = tid / slice_length;
    auto x_offset = tid - index_column * slice_length;

    for (int m = 0; m < idx_rows; m++) {
      auto index = idx_data[m * idx_cols + index_column];
      x_offset += (index < 0 ? x_shape[m] + index : index) * x_stride[m];
    }
    // The idx_data comes from a Variable that may be different at any forward
    // call. Unlike the CPU code we do not want to check the error in device
    // code (that would imply raising a trap plus always synchronization and
    // costly recovery). Still we don't want to read from unaccessible memory.
    if (x_offset < x_size) {
      y_data[tid] = x_data[x_offset];
    }
  }
}

template <typename T>
__global__ void backward(const int y_size, const T *y_grad, const int x_size,
                         T *x_grad, const int *x_shape, const int *x_stride,
                         const int *idx_data, const int idx_rows,
                         const int idx_cols) {
  NBLA_CUDA_KERNEL_LOOP(tid, y_size) {
    auto slice_length = y_size / idx_cols;
    auto index_column = tid / slice_length;
    auto x_offset = tid - index_column * slice_length;

    for (int m = 0; m < idx_rows; m++) {
      auto index = idx_data[m * idx_cols + index_column];
      x_offset += (index < 0 ? x_shape[m] + index : index) * x_stride[m];
    }
    if (x_offset < x_size) {
      atomic_add(&x_grad[x_offset], y_grad[tid]);
    }
  }
}

template <typename T>
__global__ void accum_grad(const int size, const int *idx, const T *y_grad,
                           T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { atomic_add(x_grad + idx[i], y_grad[i]); }
}
} // namespace gather_nd_cuda

template <typename T>
void GatherNdCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  GatherNd<T>::setup_impl(inputs, outputs);
  Shape_t src_meta_shape = {2 * inputs[0]->ndim()};
  src_meta_.reshape(src_meta_shape, true);
  Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto ptr = src_meta_.cast_data_and_get_pointer<int>(cpu_ctx, true);
  for (auto s : inputs[0]->shape()) {
    *ptr++ = static_cast<int>(s);
  }
  for (auto s : inputs[0]->strides()) {
    *ptr++ = static_cast<int>(s);
  }
}

template <typename T>
void GatherNdCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);
  auto src = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto idx_rows = static_cast<int>(inputs[1]->shape().at(0));
  auto idx_cols = static_cast<int>(ndi::inner_size(inputs[1]->shape(), 1));

  auto src_shape_ptr = src_meta_.get_data_pointer<int>(this->ctx_);
  auto src_stride_ptr = src_shape_ptr + inputs[0]->ndim();

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(gather_nd_cuda::forward, outputs[0]->size(),
                                 dst, inputs[0]->size(), src, src_shape_ptr,
                                 src_stride_ptr, idx, idx_rows, idx_cols);
}

template <typename T>
void GatherNdCuda<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(this->device_);

  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }

  auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);

  auto idx_rows = static_cast<int>(inputs[1]->shape().at(0));
  auto idx_cols = static_cast<int>(ndi::inner_size(inputs[1]->shape(), 1));

  auto x_shape_ptr = src_meta_.get_data_pointer<int>(this->ctx_);
  auto x_stride_ptr = x_shape_ptr + inputs[0]->ndim();

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(gather_nd_cuda::backward, outputs[0]->size(),
                                 g_y, inputs[0]->size(), g_x, x_shape_ptr,
                                 x_stride_ptr, idx, idx_rows, idx_cols);
}
} // namespace nbla
