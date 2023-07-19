// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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
#include <nbla/cuda/function/scatter_nd.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

namespace scatter_nd_cuda {

template <typename T, bool add = false>
__global__ void forward(const int x_size, const T *x_data, const int y_size,
                        T *y_data, const int *y_shape, const int *y_stride,
                        const int *idx_data, const int idx_rows,
                        const int idx_cols) {
  NBLA_CUDA_KERNEL_LOOP(tid, x_size) {
    auto slice_length = x_size / idx_cols;
    auto index_column = tid / slice_length;
    auto y_offset = tid - index_column * slice_length;

    for (int m = 0; m < idx_rows; m++) {
      auto index = idx_data[m * idx_cols + index_column];
      y_offset += (index < 0 ? y_shape[m] + index : index) * y_stride[m];
    }
    // The idx_data comes from a Variable that may be different at any forward
    // call. Unlike the CPU code we do not want to check the error in device
    // code (that would imply raising a trap plus always synchronization and
    // costly recovery). Still we don't want to write to unaccessible memory.
    if (y_offset < y_size) {
      if (add) {
        atomic_add(&y_data[y_offset], x_data[tid]);
      } else {
        // Scatter indices are supposed to be unique, i.e. not to scatter
        // different values into the same positions. Otherwise it is the last
        // update that survives which for parallel execution is unpredictable.
        y_data[y_offset] = x_data[tid];
      }
    }
  }
}

template <bool accum, typename T>
__global__ void backward(const int x_size, T *x_grad, const int y_size,
                         const T *y_grad, const int *y_shape,
                         const int *y_stride, const int *idx_data,
                         const int idx_rows, const int idx_cols) {
  NBLA_CUDA_KERNEL_LOOP(tid, x_size) {
    auto slice_length = x_size / idx_cols;
    auto index_column = tid / slice_length;
    auto y_offset = tid - index_column * slice_length;

    for (int m = 0; m < idx_rows; m++) {
      auto index = idx_data[m * idx_cols + index_column];
      y_offset += (index < 0 ? y_shape[m] + index : index) * y_stride[m];
    }
    if (y_offset < y_size) {
      x_grad[tid] = accum ? x_grad[tid] + y_grad[y_offset] : y_grad[y_offset];
    }
  }
}

template <bool accum, typename T>
__global__ void backward(const int x_size, T *x_grad, const int y_size,
                         T *y_grad, const int *y_shape, const int *y_stride,
                         const int *idx_data, const int idx_rows,
                         const int idx_cols) {
  NBLA_CUDA_KERNEL_LOOP(tid, x_size) {
    auto slice_length = x_size / idx_cols;
    auto index_column = tid / slice_length;
    auto y_offset = tid - index_column * slice_length;

    for (int m = 0; m < idx_rows; m++) {
      auto index = idx_data[m * idx_cols + index_column];
      y_offset += (index < 0 ? y_shape[m] + index : index) * y_stride[m];
    }
    if (y_offset < y_size) {
      x_grad[tid] = accum ? x_grad[tid] + y_grad[y_offset] : y_grad[y_offset];
      y_grad[y_offset] = T(0);
    }
  }
}
} // namespace scatter_nd_cuda

template <typename T>
void ScatterNdCuda<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  ScatterNd<T>::setup_impl(inputs, outputs);
  Shape_t dst_meta_shape = {2 * outputs[0]->ndim()};
  dst_meta_.reshape(dst_meta_shape, true);
  Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto ptr = dst_meta_.cast_data_and_get_pointer<int>(cpu_ctx, true);
  for (auto s : outputs[0]->shape()) {
    *ptr++ = static_cast<int>(s);
  }
  for (auto s : outputs[0]->strides()) {
    *ptr++ = static_cast<int>(s);
  }
}

template <typename T>
void ScatterNdCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(this->device_);

  if (inputs.size() < 3)
    outputs[0]->data()->zero();

  auto src = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, false);

  auto idx_rows = static_cast<int>(inputs[1]->shape().at(0));
  auto idx_cols = static_cast<int>(ndi::inner_size(inputs[1]->shape(), 1));

  auto dst_shape_ptr = dst_meta_.get_data_pointer<int>(this->ctx_);
  auto dst_stride_ptr = dst_shape_ptr + outputs[0]->ndim();

  auto kernel = this->add_ ? scatter_nd_cuda::forward<Tcu, true>
                           : scatter_nd_cuda::forward<Tcu, false>;

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, inputs[0]->size(), src,
                                 outputs[0]->size(), dst, dst_shape_ptr,
                                 dst_stride_ptr, idx, idx_rows, idx_cols);
}

template <typename T>
void ScatterNdCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  cuda_set_device(this->device_);

  auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  auto idx = inputs[1]->get_data_pointer<int>(this->ctx_);

  auto idx_rows = static_cast<int>(inputs[1]->shape().at(0));
  auto idx_cols = static_cast<int>(ndi::inner_size(inputs[1]->shape(), 1));

  auto y_shape_ptr = dst_meta_.get_data_pointer<int>(this->ctx_);
  auto y_stride_ptr = y_shape_ptr + outputs[0]->ndim();

  if (inputs.size() < 3) {
    // Because input[0] data is scattered into a new output variable during
    // forward, output[0] gradient values from scatter indices are propagated
    // back to input[0] gradient.
    auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(scatter_nd_cuda::backward<true>,
                                     inputs[0]->size(), g_x, outputs[0]->size(),
                                     g_y, y_shape_ptr, y_stride_ptr, idx,
                                     idx_rows, idx_cols);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(scatter_nd_cuda::backward<false>,
                                     inputs[0]->size(), g_x, outputs[0]->size(),
                                     g_y, y_shape_ptr, y_stride_ptr, idx,
                                     idx_rows, idx_cols);
    }
  } else {
    // Because input[0] data is scattered into the data of input[2] (the input
    // parameter named `out`) inplaced with output[0], the gradient values of
    // output[0] that belong to the scatter indices are propagated back to the
    // input[0] gradient and set to 0 (masked) in the grad array of output[0].
    auto g_y = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(scatter_nd_cuda::backward<true>,
                                     inputs[0]->size(), g_x, outputs[0]->size(),
                                     g_y, y_shape_ptr, y_stride_ptr, idx,
                                     idx_rows, idx_cols);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(scatter_nd_cuda::backward<false>,
                                     inputs[0]->size(), g_x, outputs[0]->size(),
                                     g_y, y_shape_ptr, y_stride_ptr, idx,
                                     idx_rows, idx_cols);
    }
  }
}
} // namespace nbla
