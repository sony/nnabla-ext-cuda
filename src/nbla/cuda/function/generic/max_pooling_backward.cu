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
#include <nbla/cuda/function/max_pooling_backward.hpp>
#include <nbla/cuda/utils/nd_index.hpp>
#include <nbla/variable.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace nbla {

template <typename T>
void MaxPoolingBackwardCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  MaxPoolingBackward<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void MaxPoolingBackwardCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  NBLA_ERROR(error_code::not_implemented,
             "Do not call MaxPoolingBackward::forward. \n"
             "This is the temporal function to support the double backward of "
             "the max pooling. \n"
             "Directly call the backward method.");
}

// TODO: Optimize
template <typename T, int NDIM, bool accum = true>
__global__ void kernel_max_pooling_2d_double_backward(
    const Size_t y_size, const T *x, const T *g_dx, T *g_dy,
    const int64_t *x_shape, const int64_t *x_stride, const int64_t *y_stride,
    const int *kernel, const int *stride, const int *pad,
    const bool channel_last) {
  auto hdim = channel_last ? NDIM - 3 : NDIM - 2;
  auto wdim = channel_last ? NDIM - 2 : NDIM - 1;
  NBLA_CUDA_KERNEL_LOOP(idx, y_size) {
    // 1. NdIndex
    NdIndex<NDIM> nd_index = device_flat2nd<NDIM>(idx, y_stride);

    // 2. Create pool indices
    int64_t h0 = nd_index.nd_idx[hdim];
    int64_t w0 = nd_index.nd_idx[wdim];
    auto pool_h_start = max(h0 * stride[0] - pad[0], (int64_t)0);
    auto pool_h_end = min(h0 * stride[0] + kernel[0] - pad[0], x_shape[hdim]);
    auto pool_w_start = max(w0 * stride[1] - pad[1], (int64_t)0);
    auto pool_w_end = min(w0 * stride[1] + kernel[1] - pad[1], x_shape[wdim]);

    // 3. Initial value
    nd_index.nd_idx[hdim] = pool_h_start;
    nd_index.nd_idx[wdim] = pool_w_start;
    auto max_idx = device_nd2flat<NDIM>(nd_index, x_stride);
    auto max_val = x[max_idx];

    // 4. Find max index
    for (int h = pool_h_start; h < pool_h_end; h++) {
      for (int w = pool_w_start; w < pool_w_end; w++) {
        nd_index.nd_idx[hdim] = h;
        nd_index.nd_idx[wdim] = w;
        auto idx_r = device_nd2flat<NDIM>(nd_index, x_stride);
        auto val = x[idx_r];
        if (val > max_val) {
          max_val = val;
          max_idx = idx_r;
        }
      }
    }

    // 5. Double backward is table look-up with max index
    g_dy[idx] = accum ? g_dy[idx] + g_dx[max_idx] : g_dx[max_idx];
  }
}

template <typename T, int NDIM, bool accum = true>
__global__ void kernel_max_pooling_3d_double_backward(
    const Size_t y_size, const T *x, const T *g_dx, T *g_dy,
    const int64_t *x_shape, const int64_t *x_stride, const int64_t *y_stride,
    const int *kernel, const int *stride, const int *pad,
    const bool channel_last) {
  auto ddim = channel_last ? NDIM - 4 : NDIM - 3;
  auto hdim = channel_last ? NDIM - 3 : NDIM - 2;
  auto wdim = channel_last ? NDIM - 2 : NDIM - 1;
  NBLA_CUDA_KERNEL_LOOP(idx, y_size) {
    // 1. NdIndex
    NdIndex<NDIM> nd_index = device_flat2nd<NDIM>(idx, y_stride);

    // 2. Create pool indices
    int64_t d0 = nd_index.nd_idx[ddim];
    int64_t h0 = nd_index.nd_idx[hdim];
    int64_t w0 = nd_index.nd_idx[wdim];
    auto pool_d_start = max(d0 * stride[0] - pad[0], (int64_t)0);
    auto pool_d_end = min(d0 * stride[0] + kernel[0] - pad[0], x_shape[ddim]);
    auto pool_h_start = max(h0 * stride[1] - pad[1], (int64_t)0);
    auto pool_h_end = min(h0 * stride[1] + kernel[1] - pad[1], x_shape[hdim]);
    auto pool_w_start = max(w0 * stride[2] - pad[2], (int64_t)0);
    auto pool_w_end = min(w0 * stride[2] + kernel[2] - pad[2], x_shape[wdim]);

    // 3. Initial value
    nd_index.nd_idx[ddim] = pool_d_start;
    nd_index.nd_idx[hdim] = pool_h_start;
    nd_index.nd_idx[wdim] = pool_w_start;
    auto max_idx = device_nd2flat<NDIM>(nd_index, x_stride);
    auto max_val = x[max_idx];

    // 4. Find max index
    for (int d = pool_d_start; d < pool_d_end; d++) {
      for (int h = pool_h_start; h < pool_h_end; h++) {
        for (int w = pool_w_start; w < pool_w_end; w++) {
          nd_index.nd_idx[ddim] = d;
          nd_index.nd_idx[hdim] = h;
          nd_index.nd_idx[wdim] = w;
          auto idx_r = device_nd2flat<NDIM>(nd_index, x_stride);
          auto val = x[idx_r];
          if (val > max_val) {
            max_val = val;
            max_idx = idx_r;
          }
        }
      }
    }

    // 5. Double backward is table look-up with max index
    g_dy[idx] = accum ? g_dy[idx] + g_dx[max_idx] : g_dx[max_idx];
  }
}

template <typename T>
void MaxPoolingBackwardCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  if (propagate_down[0]) {
    if (!accum[0]) {
      inputs[0]->grad()->zero();
    }
  }
  if (propagate_down[1]) {
    auto y_size = inputs[1]->size();
    const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
    const Tcu *g_dx = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
    Tcu *g_dy = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
    thrust::device_vector<int64_t> d_vec_x_shape(inputs[0]->shape());
    thrust::device_vector<int64_t> d_vec_x_stride(inputs[0]->strides());
    thrust::device_vector<int64_t> d_vec_y_stride(inputs[1]->strides());
    thrust::device_vector<int> d_vec_kernel(this->kernel_);
    thrust::device_vector<int> d_vec_stride(this->stride_);
    thrust::device_vector<int> d_vec_pad(this->pad_);
    auto x_shape = thrust::raw_pointer_cast(d_vec_x_shape.data());
    auto x_stride = thrust::raw_pointer_cast(d_vec_x_stride.data());
    auto y_stride = thrust::raw_pointer_cast(d_vec_y_stride.data());
    auto kernel = thrust::raw_pointer_cast(d_vec_kernel.data());
    auto stride = thrust::raw_pointer_cast(d_vec_stride.data());
    auto pad = thrust::raw_pointer_cast(d_vec_pad.data());
    auto channel_last = this->channel_last_;

    auto ndim = inputs[0]->shape().size();
    if (this->kernel_.size() == 2) {
      if (ndim == 4) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 4, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 4, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      } else if (ndim == 5) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 5, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 5, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      } else if (ndim == 6) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 6, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_2d_double_backward<Tcu, 6, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      }
    } else if (this->kernel_.size() == 3) {
      if (ndim == 5) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 5, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 5, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      } else if (ndim == 6) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 6, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 6, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      } else if (ndim == 7) {
        if (accum[1]) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 7, true>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        } else {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_max_pooling_3d_double_backward<Tcu, 7, false>), y_size, x,
              g_dx, g_dy, x_shape, x_stride, y_stride, kernel, stride, pad,
              channel_last);
        }
      }
    }
  }
}
}
