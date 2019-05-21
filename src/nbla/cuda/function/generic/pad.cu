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
#include <nbla/cuda/function/pad.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

using cuda::Index_t;

struct AxisParam {
  Index_t x_stride;
  Index_t y_stride;
  Index_t y_shape;
  struct {
    Index_t first;
    Index_t second;
  } pad;
};

template <int DIMENSIONS>
__inline__ __device__ void d_init_index_map(const Index_t y_idx,
                                            Index_t *idx_map, const int ndim,
                                            const AxisParam *params) {
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  Index_t y_tmp = y_idx;
  Index_t x_idx = 0;

#pragma unroll
  for (int axis = 0; axis < NDIM; axis++) {
    const auto &param = params[axis];
    const auto axis_idx = y_tmp / param.y_stride;
    y_tmp -= axis_idx * param.y_stride;

    if ((axis_idx < param.pad.first) ||
        (axis_idx >= param.y_shape - param.pad.second)) {
      return;
    }
    x_idx += (axis_idx - param.pad.first) * param.x_stride;
  }
  idx_map[y_idx] = x_idx;
}

template <int DIMENSIONS = 0>
__global__ void init_index_map(const Index_t size, Index_t *idx_map,
                               const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_init_index_map<DIMENSIONS>(i, idx_map, ndim, shared);
  }
}

namespace pad_constant_impl {

template <typename T, int DIMENSIONS>
__inline__ __device__ void d_pad_forward(const Index_t y_idx, const T *x, T *y,
                                         const int ndim,
                                         const AxisParam *params, const T val) {
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  Index_t y_tmp = y_idx;
  Index_t x_idx = 0;

#pragma unroll
  for (int axis = 0; axis < NDIM; axis++) {
    const auto &param = params[axis];
    const auto axis_idx = y_tmp / param.y_stride;
    y_tmp -= axis_idx * param.y_stride;

    if ((axis_idx < param.pad.first) ||
        (axis_idx >= param.y_shape - param.pad.second)) {
      y[y_idx] = val;
      return;
    }
    x_idx += (axis_idx - param.pad.first) * param.x_stride;
  }
  y[y_idx] = x[x_idx];
}

template <typename T, int DIMENSIONS = 0>
__global__ void pad_forward(const Index_t size, const T *x, T *y,
                            const int ndim, const AxisParam *params,
                            const T constant_value) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_forward<T, DIMENSIONS>(i, x, y, ndim, shared, constant_value);
  }
}

template <typename T, bool ACCUMULATE, int DIMENSIONS>
__inline__ __device__ void d_pad_backward(const Index_t y_idx, const T *dy,
                                          T *dx, const int ndim,
                                          const AxisParam *params) {
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  Index_t y_tmp = y_idx;
  Index_t x_idx = 0;

#pragma unroll
  for (int axis = 0; axis < NDIM; axis++) {
    const auto &param = params[axis];
    const auto axis_idx = y_tmp / param.y_stride;
    y_tmp -= axis_idx * param.y_stride;

    if ((axis_idx < param.pad.first) ||
        (axis_idx >= param.y_shape - param.pad.second)) {
      return;
    }
    x_idx += (axis_idx - param.pad.first) * param.x_stride;
  }
  dx[x_idx] = ACCUMULATE ? dx[x_idx] + dy[y_idx] : dy[y_idx];
}

template <typename T, int DIMENSIONS = 0, bool ACCUMULATE = false>
__global__ void pad_backward(const Index_t size, const T *dy, T *dx,
                             const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_backward<T, ACCUMULATE, DIMENSIONS>(i, dy, dx, ndim, shared);
  }
}

} // namespace pad_constant_impl

namespace pad_reflect_impl {

__inline__ __device__ Index_t reflect_index(Index_t idx, Index_t len) {
  return len > 0 ? std::abs(((idx / len) & 1) * len - (idx % len)) : 0;
}

__inline__ __device__ void d_pad_index_map(const Index_t dst_idx,
                                           Index_t *idx_map, const int ndim,
                                           const int axis,
                                           const AxisParam *params) {
  // This function runs for each idx_map element and copies the index
  // of a reflected output element to the corresponding location in
  // idx_map. The idx_map has the same size as the output array.
  Index_t dst_tmp = dst_idx;
  Index_t src_idx = 0;
  Index_t axis_idx = 0;

  for (int ax = 0; ax < axis; ax++) {
    axis_idx = dst_tmp / params[ax].y_stride;
    dst_tmp = dst_tmp - axis_idx * params[ax].y_stride;
    src_idx += axis_idx * params[ax].y_stride;
  }
  axis_idx = dst_tmp / params[axis].y_stride;
  dst_tmp = dst_tmp - axis_idx * params[axis].y_stride;

  const auto pad_sum = params[axis].pad.first + params[axis].pad.second;
  const auto src_len = params[axis].y_shape - pad_sum;

  if (axis_idx < params[axis].pad.first) {
    const auto p = params[axis].pad.first;
    const auto r = reflect_index(p - axis_idx, src_len - 1);
    src_idx += (p + r) * params[axis].y_stride;
    for (int ax = axis + 1; ax < ndim; ax++) {
      axis_idx = dst_tmp / params[ax].y_stride;
      dst_tmp = dst_tmp - axis_idx * params[ax].y_stride;
      src_idx += axis_idx * params[ax].y_stride;
    }
    idx_map[dst_idx] = idx_map[src_idx];
    return;
  }

  if (axis_idx >= params[axis].y_shape - params[axis].pad.second) {
    const auto p = params[axis].pad.first + src_len;
    const auto r = reflect_index(axis_idx - p + 1, src_len - 1);
    src_idx += (p - r - 1) * params[axis].y_stride;
    for (int ax = axis + 1; ax < ndim; ax++) {
      axis_idx = dst_tmp / params[ax].y_stride;
      dst_tmp = dst_tmp - axis_idx * params[ax].y_stride;
      src_idx += axis_idx * params[ax].y_stride;
    }
    idx_map[dst_idx] = idx_map[src_idx];
    return;
  }
}

__global__ void pad_index_map(const Index_t size, Index_t *idx, const int ndim,
                              const int axis, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  if (threadIdx.x < ndim * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_index_map(i, idx, ndim, axis, shared);
  }
}

template <typename T>
__global__ void pad_forward(const Index_t size, const T *x, T *y,
                            const Index_t *idx_map) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { y[i] = x[idx_map[i]]; }
}

template <typename T>
__global__ void pad_backward(const Index_t size, const T *dy, T *dx,
                             const Index_t *idx_map) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { atomic_add(&dx[idx_map[i]], dy[i]); }
}

} // namespace pad_reflect_impl

template <typename T>
void PadCuda<T>::setup_impl(const Variables &inputs, const Variables &outputs) {

  Pad<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  Variable &x_var = *inputs[0];
  Variable &y_var = *outputs[0];

  std::vector<AxisParam> h_params;
  h_params.reserve(this->padding_.size());
  for (int axis = 0; axis < this->padding_.size(); axis++) {
    AxisParam axis_param;
    axis_param.x_stride = this->x_stride_.at(axis);
    axis_param.y_stride = this->y_stride_.at(axis);
    axis_param.y_shape = this->y_shape_.at(axis);
    axis_param.pad.first = this->padding_.at(axis).first;
    axis_param.pad.second = this->padding_.at(axis).second;
    h_params.push_back(axis_param);
  }
  auto bytes = h_params.size() * sizeof(AxisParam);
  auto array = new CudaCachedArray(bytes, get_dtype<char>(), this->ctx_);
  auto d_params = array->pointer<AxisParam>();
  NBLA_CUDA_CHECK(
      cudaMemcpy(d_params, h_params.data(), bytes, cudaMemcpyHostToDevice));
  this->parameter_memory_ = std::unique_ptr<CudaCachedArray>(std::move(array));
}

template <typename T>
void PadCuda<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  cuda_set_device(this->device_);

  Variable &x_var = *inputs[0];
  Variable &y_var = *outputs[0];

  const auto y_size = y_var.size();
  const auto ndim = this->padding_.size();

  auto x = x_var.get_data_pointer<Tcu>(this->ctx_);
  auto y = y_var.cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto threads = 128;
  auto blocks = cuda_get_blocks_by_size(y_var.size());
  auto shared = this->parameter_memory_->size();
  auto params = this->parameter_memory_->template pointer<AxisParam>();

  if (this->pad_mode_ == this->PAD_CONSTANT) {
    using pad_constant_impl::pad_forward;
    auto cvalue = this->constant_value_;
    void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                   const AxisParam *, const Tcu);
    if (ndim == 1) {
      kernel = pad_forward<Tcu, 1>;
    } else if (ndim == 2) {
      kernel = pad_forward<Tcu, 2>;
    } else if (ndim == 3) {
      kernel = pad_forward<Tcu, 3>;
    } else if (ndim == 4) {
      kernel = pad_forward<Tcu, 4>;
    } else {
      kernel = pad_forward<Tcu>;
    }
    kernel<<<blocks, threads, shared>>>(y_size, x, y, ndim, params, cvalue);
    NBLA_CUDA_KERNEL_CHECK();
  }

  else if (this->pad_mode_ == this->PAD_REFLECT) {
    using namespace pad_reflect_impl;
    Variable &idx_map = this->index_map_;
    auto idx = idx_map.cast_data_and_get_pointer<Index_t>(this->ctx_, false);
    void (*kernel)(const Index_t, Index_t *, const int, const AxisParam *);
    if (ndim == 1) {
      kernel = init_index_map<1>;
    } else if (ndim == 2) {
      kernel = init_index_map<2>;
    } else if (ndim == 3) {
      kernel = init_index_map<3>;
    } else if (ndim == 4) {
      kernel = init_index_map<4>;
    } else {
      kernel = init_index_map<>;
    }
    kernel<<<blocks, threads, shared>>>(y_size, idx, ndim, params);
    NBLA_CUDA_KERNEL_CHECK();
    // Padding the index map must be done with individual kernel
    // launches to synchronize index values which become source of
    // padding for the next outer axis.
    for (int axis = ndim - 1; axis >= 0; axis--) {
      auto kernel = pad_index_map;
      kernel<<<blocks, threads, shared>>>(y_size, idx, ndim, axis, params);
      NBLA_CUDA_KERNEL_CHECK();
    }
    // Perform y[i] = x[idx[i]] for all i in y_size
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(pad_forward, y_size, x, y, idx);
  }
}

template <typename T>
void PadCuda<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum_gradient) {
  if (propagate_down[0]) {
    cuda_set_device(this->device_);
    auto accum = accum_gradient[0];

    Variable &x_var = *inputs[0];
    Variable &y_var = *outputs[0];

    const auto ndim = this->padding_.size();
    auto dy = y_var.get_grad_pointer<Tcu>(this->ctx_);

    if (this->pad_mode_ == this->PAD_CONSTANT) {
      using namespace pad_constant_impl;
      auto dx = x_var.cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum);
      auto threads = 128;
      auto blocks = cuda_get_blocks_by_size(y_var.size());
      auto shared = this->parameter_memory_->size();
      auto params = this->parameter_memory_->template pointer<AxisParam>();
      void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                     const AxisParam *);
      if (ndim == 1) {
        kernel = accum ? pad_backward<Tcu, 1, true> : pad_backward<Tcu, 1>;
      } else if (ndim == 2) {
        kernel = accum ? pad_backward<Tcu, 2, true> : pad_backward<Tcu, 2>;
      } else if (ndim == 3) {
        kernel = accum ? pad_backward<Tcu, 3, true> : pad_backward<Tcu, 3>;
      } else if (ndim == 4) {
        kernel = accum ? pad_backward<Tcu, 4, true> : pad_backward<Tcu, 4>;
      } else {
        kernel = accum ? pad_backward<Tcu, 0, true> : pad_backward<Tcu>;
      }
      kernel<<<blocks, threads, shared>>>(y_var.size(), dy, dx, ndim, params);
      NBLA_CUDA_KERNEL_CHECK();
    }

    else if (this->pad_mode_ == this->PAD_REFLECT) {
      if (!accum) {
        x_var.grad()->zero();
      }
      Variable &idx_map = this->index_map_;
      auto idx = idx_map.get_data_pointer<Index_t>(this->ctx_);
      auto dx = x_var.cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      auto backward = pad_reflect_impl::pad_backward<Tcu>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward, y_var.size(), dy, dx, idx);
    }
  }
}

} // namespace nbla
