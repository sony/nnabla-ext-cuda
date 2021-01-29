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

template <typename T, int DIMENSIONS>
__inline__ __device__ void d_pad_reflect_forward(const Index_t y_idx, const T *x, T *y,
                                         const int ndim,
                                         const AxisParam *params) {
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  Index_t y_tmp = y_idx;
  Index_t x_idx = 0;

#pragma unroll
  for (int axis = 0; axis < NDIM; axis++) {
    const auto &param = params[axis];
    const auto axis_idx = y_tmp / param.y_stride;
    y_tmp -= axis_idx * param.y_stride;

    const auto src_len = param.y_shape - param.pad.first - param.pad.second;
    Index_t src_axis_idx = std::abs(axis_idx - param.pad.first);

    const auto src_axis_reflect_idx = reflect_index(src_axis_idx, src_len - 1);
    x_idx += src_axis_reflect_idx * param.x_stride;

  }
  y[y_idx] = x[x_idx];
}

template <typename T, int DIMENSIONS = 0>
__global__ void pad_reflect_forward(const Index_t size, const T *x, T *y,
                            const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_reflect_forward<T, DIMENSIONS>(i, x, y, ndim, shared);
  }
}

template <typename T, int DIMENSIONS>
__inline__ __device__ void d_pad_reflect_backward(const Index_t y_idx, const T *dy,
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

    const auto dst_len = param.y_shape - param.pad.first - param.pad.second;
    Index_t dst_axis_idx = std::abs(axis_idx - param.pad.first);

    const auto dst_axis_reflect_idx = reflect_index(dst_axis_idx, dst_len - 1);
    x_idx += dst_axis_reflect_idx * param.x_stride;
  }
  atomic_add(&dx[x_idx], dy[y_idx]);
}

template <typename T, int DIMENSIONS = 0>
__global__ void pad_reflect_backward(const Index_t size, const T *dy, T *dx,
                             const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_reflect_backward<T, DIMENSIONS>(i, dy, dx, ndim, shared);
  }
}

} // namespace pad_reflect_impl

namespace pad_repeat_impl {

template <typename T, int DIMENSIONS>
__inline__ __device__ void d_pad_repeat_forward(const Index_t y_idx, const T *x, T *y,
                                         const int ndim,
                                         const AxisParam *params) {
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  Index_t y_tmp = y_idx;
  Index_t x_idx = 0;

#pragma unroll
  for (int axis = 0; axis < NDIM; axis++) {
    const auto &param = params[axis];
    const auto axis_idx = y_tmp / param.y_stride;
    y_tmp -= axis_idx * param.y_stride;

    int src_max_idx = param.y_shape - param.pad.first - param.pad.second - 1;
    x_idx += min(src_max_idx , max(0, static_cast<int>(axis_idx - param.pad.first))) * param.x_stride;
  }
  y[y_idx] = x[x_idx];
}

template <typename T, int DIMENSIONS = 0>
__global__ void pad_repeat_forward(const Index_t size, const T *x, T *y,
                            const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_repeat_forward<T, DIMENSIONS>(i, x, y, ndim, shared);
  }
}

template <typename T, int DIMENSIONS>
__inline__ __device__ void d_pad_repeat_backward(const Index_t y_idx, const T *dy,
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

    int dst_max_idx = param.y_shape - param.pad.first - param.pad.second - 1;
    x_idx += min(dst_max_idx , max(0, static_cast<int>(axis_idx - param.pad.first))) * param.x_stride;
  }
  atomic_add(&dx[x_idx], dy[y_idx]);
}

template <typename T, int DIMENSIONS = 0>
__global__ void pad_repeat_backward(const Index_t size, const T *dy, T *dx,
                             const int ndim, const AxisParam *params) {
  extern __shared__ AxisParam shared[];
  const int NDIM = DIMENSIONS > 0 ? DIMENSIONS : ndim;
  if (threadIdx.x < NDIM * sizeof(AxisParam) / sizeof(int)) {
    auto tmp = reinterpret_cast<const int *>(params)[threadIdx.x];
    reinterpret_cast<int *>(shared)[threadIdx.x] = tmp;
  }
  __syncthreads();
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    d_pad_repeat_backward<T, DIMENSIONS>(i, dy, dx, ndim, shared);
  }
}

} // namespace pad_repeat_impl

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
  this->parameter_memory_.reshape(Shape_t{static_cast<Size_t>(bytes)}, true);
  auto d_params =
      this->parameter_memory_.cast(get_dtype<char>(), this->ctx_, true)
          ->template pointer<AxisParam>();
  NBLA_CUDA_CHECK(
      cudaMemcpy(d_params, h_params.data(), bytes, cudaMemcpyHostToDevice));
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
  auto shared = this->parameter_memory_.size();
  auto params = this->parameter_memory_.get(get_dtype<char>(), this->ctx_)
                    ->template const_pointer<AxisParam>();

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
    void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                   const AxisParam *);
    if (ndim == 1) {
      kernel = pad_reflect_forward<Tcu, 1>;
    } else if (ndim == 2) {
      kernel = pad_reflect_forward<Tcu, 2>;
    } else if (ndim == 3) {
      kernel = pad_reflect_forward<Tcu, 3>;
    } else if (ndim == 4) {
      kernel = pad_reflect_forward<Tcu, 4>;
    } else {
      kernel = pad_reflect_forward<Tcu>;
    }
    kernel<<<blocks, threads, shared>>>(y_size, x, y, ndim, params);
    NBLA_CUDA_KERNEL_CHECK();
  }
  else if (this->pad_mode_ == this->PAD_REPEAT) {
    using pad_repeat_impl::pad_repeat_forward;
    void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                   const AxisParam *);
    if (ndim == 1) {
      kernel = pad_repeat_forward<Tcu, 1>;
    } else if (ndim == 2) {
      kernel = pad_repeat_forward<Tcu, 2>;
    } else if (ndim == 3) {
      kernel = pad_repeat_forward<Tcu, 3>;
    } else if (ndim == 4) {
      kernel = pad_repeat_forward<Tcu, 4>;
    } else {
      kernel = pad_repeat_forward<Tcu>;
    }
    kernel<<<blocks, threads, shared>>>(y_size, x, y, ndim, params);
    NBLA_CUDA_KERNEL_CHECK();
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
      auto shared = this->parameter_memory_.size();
      auto params = this->parameter_memory_.get(get_dtype<char>(), this->ctx_)
                        ->template const_pointer<AxisParam>();
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
      using namespace pad_reflect_impl;
      if (!accum) {
        x_var.grad()->zero();
      }
      auto dx = x_var.cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      auto threads = 128;
      auto blocks = cuda_get_blocks_by_size(y_var.size());
      auto shared = this->parameter_memory_.size();
      auto params = this->parameter_memory_.get(get_dtype<char>(), this->ctx_)
                        ->template const_pointer<AxisParam>();
      void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                     const AxisParam *);
      if (ndim == 1) {
        kernel = pad_reflect_backward<Tcu, 1>;
      } else if (ndim == 2) {
        kernel = pad_reflect_backward<Tcu, 2>;
      } else if (ndim == 3) {
        kernel = pad_reflect_backward<Tcu, 3>;
      } else if (ndim == 4) {
        kernel = pad_reflect_backward<Tcu, 4>;
      } else {
        kernel = pad_reflect_backward<Tcu>;
      }
      kernel<<<blocks, threads, shared>>>(y_var.size(), dy, dx, ndim, params);
      NBLA_CUDA_KERNEL_CHECK();
    }
    else if (this->pad_mode_ == this->PAD_REPEAT) {
      using namespace pad_repeat_impl;
      if (!accum) {
        x_var.grad()->zero();
      }
      auto dx = x_var.cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      auto threads = 128;
      auto blocks = cuda_get_blocks_by_size(y_var.size());
      auto shared = this->parameter_memory_.size();
      auto params = this->parameter_memory_.get(get_dtype<char>(), this->ctx_)
                        ->template const_pointer<AxisParam>();
      void (*kernel)(const Index_t, const Tcu *, Tcu *, const int,
                     const AxisParam *);
      if (ndim == 1) {
        kernel = pad_repeat_backward<Tcu, 1>;
      } else if (ndim == 2) {
        kernel = pad_repeat_backward<Tcu, 2>;
      } else if (ndim == 3) {
        kernel = pad_repeat_backward<Tcu, 3>;
      } else if (ndim == 4) {
        kernel = pad_repeat_backward<Tcu, 4>;
      } else {
        kernel = pad_repeat_backward<Tcu>;
      }
      kernel<<<blocks, threads, shared>>>(y_var.size(), dy, dx, ndim, params);
      NBLA_CUDA_KERNEL_CHECK();
    }
  }
}

} // namespace nbla
