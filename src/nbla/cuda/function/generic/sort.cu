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
#include <nbla/cuda/function/sort.hpp>
#include <nbla/variable.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#if THRUST_VERSION < 100904 || !defined(THRUST_CPP11) ||                       \
    !defined(THRUST_MODERN_GCC)
#include <thrust/sort.h>
#else
#include <thrust/async/sort.h>
#endif

namespace nbla {

namespace sort_impl {

template <typename T> struct Compare {
  const T *data;
  const size_t stride;
  const bool reverse;

  __host__ __device__ Compare(const T *data, const size_t stride,
                              const bool reverse)
      : data(data), stride(stride), reverse(reverse) {}

  __host__ __device__ bool operator()(size_t i1, size_t i2) const {
    return data[i1 * stride] < data[i2 * stride] != reverse;
  }
};

__global__ void make_sequence(const size_t size, size_t *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dst[i] = static_cast<size_t>(i); }
}

__global__ void copy_index(const size_t size, const size_t stride,
                           const size_t *src, size_t *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dst[i * stride] = src[i]; }
}

template <typename T>
__global__ void copy_value(const size_t size, const size_t stride, const T *src,
                           const size_t *idx, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dst[i * stride] = src[idx[i * stride] * stride];
  }
}

template <typename T>
__global__ void add_grad(const size_t size, const size_t stride, const T *src,
                         const size_t *idx, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dst[i * stride] += src[idx[i * stride] * stride];
  }
}

template <typename T>
__global__ void set_grad(const size_t size, const size_t stride, const T *src,
                         const size_t *idx, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dst[i * stride] = src[idx[i * stride] * stride];
  }
}

} // namespace sort_impl

template <typename T>
void SortCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
#if THRUST_VERSION < 100904 || !defined(THRUST_CPP11) ||                       \
    !defined(THRUST_MODERN_GCC)
  using thrust::sort;
#else
  using thrust::async::sort;
#endif

  using namespace sort_impl;
  cuda_set_device(this->device_);

  const auto &ctx = this->ctx_;
  const auto &shape = inputs[0]->shape();

  Variable &sort_index_var = this->sort_index;
  Variable &temp_index_var = this->temp_index;
  auto sort_index_raw = sort_index_var.cast_data_and_get_pointer<size_t>(ctx);
  auto temp_index_raw = temp_index_var.cast_data_and_get_pointer<size_t>(ctx);
  auto temp_index_ptr = thrust::device_pointer_cast(temp_index_raw);
  auto x_data = inputs[0]->get_data_pointer<Tcu>(ctx);

  auto outer_x_raw = x_data;
  auto outer_i_raw = sort_index_raw;
  auto stride = this->inner_size;

  cudaStream_t stream;
  NBLA_CUDA_CHECK(cudaStreamCreate(&stream));

  while (outer_x_raw < x_data + this->total_size) {
    auto inner_x_raw = outer_x_raw;
    auto inner_i_raw = outer_i_raw;

    while (inner_x_raw < outer_x_raw + this->inner_size) {
      auto size = temp_index_var.size();
      auto compare = Compare<Tcu>(inner_x_raw, stride, this->reverse);
      NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(make_sequence, stream, size,
                                        temp_index_raw);
      (void)sort(thrust::cuda::par.on(stream), temp_index_ptr,
                 temp_index_ptr + size, compare);
      NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(copy_index, stream, shape[this->axis],
                                        stride, temp_index_raw, inner_i_raw);
      inner_x_raw++;
      inner_i_raw++;
    }
    outer_x_raw += this->outer_size;
    outer_i_raw += this->outer_size;
  }

  NBLA_CUDA_CHECK(cudaStreamDestroy(stream));

  if (!this->only_index) {
    auto y_data = outputs[0]->cast_data_and_get_pointer<Tcu>(ctx, true);
    auto outer_x_raw = x_data;
    auto outer_y_raw = y_data;
    auto outer_i_raw = sort_index_raw;
    auto stride = this->inner_size;

    while (outer_x_raw < x_data + this->total_size) {
      auto inner_x_raw = outer_x_raw;
      auto inner_y_raw = outer_y_raw;
      auto inner_i_raw = outer_i_raw;

      while (inner_x_raw < outer_x_raw + this->inner_size) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_value, shape[this->axis], stride,
                                       inner_x_raw, inner_i_raw, inner_y_raw);
        inner_x_raw++;
        inner_y_raw++;
        inner_i_raw++;
      }
      outer_x_raw += this->outer_size;
      outer_y_raw += this->outer_size;
      outer_i_raw += this->outer_size;
    }
  }

  if (this->with_index || this->only_index) {
    Variable *out_var = this->only_index ? outputs[0] : outputs[1];
    auto out_arr = out_var->data()->cast(get_dtype<size_t>(), ctx, true);
    auto idx_buf = this->sort_index.data()->get(get_dtype<int>(), ctx);
    out_arr->copy_from(idx_buf);
  }
}

template <typename T>
void SortCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  using namespace sort_impl;

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const auto &ctx = this->ctx_;
  const auto &shape = inputs[0]->shape();

  Variable &sort_index_var = this->sort_index;
  auto sort_index_raw = sort_index_var.cast_data_and_get_pointer<size_t>(ctx);
  auto x_grad = inputs[0]->cast_grad_and_get_pointer<Tcu>(ctx, !accum[0]);
  auto y_grad = outputs[0]->get_grad_pointer<Tcu>(ctx);

  auto outer_x_raw = x_grad;
  auto outer_y_raw = y_grad;
  auto outer_i_raw = sort_index_raw;
  auto stride = this->inner_size;

  while (outer_x_raw < x_grad + this->total_size) {
    auto inner_x_raw = outer_x_raw;
    auto inner_y_raw = outer_y_raw;
    auto inner_i_raw = outer_i_raw;

    while (inner_y_raw < outer_y_raw + this->inner_size) {
      if (accum[0]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(add_grad, shape[this->axis], stride,
                                       inner_y_raw, inner_i_raw, inner_x_raw);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_grad, shape[this->axis], stride,
                                       inner_y_raw, inner_i_raw, inner_x_raw);
      }
      inner_x_raw++;
      inner_y_raw++;
      inner_i_raw++;
    }
    outer_x_raw += this->outer_size;
    outer_y_raw += this->outer_size;
    outer_i_raw += this->outer_size;
  }
}

} // namespace nbla
