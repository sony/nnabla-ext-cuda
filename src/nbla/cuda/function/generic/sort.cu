// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <cub/cub.cuh>
#include <nbla/function/transpose.hpp>

// define cub::NumericTraits for nbla::HalfCuda to use cub functions.
template <>
struct cub::NumericTraits<nbla::HalfCuda>
    : cub::BaseTraits<FLOATING_POINT, true, false, unsigned short,
                      nbla::HalfCuda> {};

namespace nbla {

namespace sort_impl {

template <typename T> struct CompareGreater {
  const T *data;
  const size_t stride;

  __host__ __device__ CompareGreater(const T *data, const size_t stride)
      : data(data), stride(stride) {}

  __host__ __device__ bool operator()(size_t i1, size_t i2) const {
    return data[i1 * stride] > data[i2 * stride];
  }
};

template <typename T> struct CompareLess {
  const T *data;
  const size_t stride;

  __host__ __device__ CompareLess(const T *data, const size_t stride)
      : data(data), stride(stride) {}

  __host__ __device__ bool operator()(size_t i1, size_t i2) const {
    return data[i1 * stride] < data[i2 * stride];
  }
};

__global__ void make_sequence(const size_t size, size_t *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dst[i] = static_cast<size_t>(i); }
}

__global__ void make_index(const int total_size, const int segment_size,
                           int *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, total_size) {
    dst[i] = static_cast<size_t>(i % segment_size);
  }
}

__global__ void make_offsets(const int size, const int sort_size, int *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dst[i] = static_cast<size_t>(i * sort_size);
  }
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
void SortCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Sort<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  // Use cub sort for most cases. Slower thrust implementation is left for
  // 64 bits indexing cases since cub does not support it.
  use_cub_ = inputs[0]->size() < std::numeric_limits<int>::max();
  if (use_cub_) {
    // Cub needs contiguous memory layout for data to sort.
    // Here, transpose functions are used for re-arranging memory layout before
    // and after cub sort call.
    need_transpose_ = this->inner_size != 1;
    if (need_transpose_) {
      // Create transpose axes.
      // e.g.)
      // input_ndim = 6, sort_axis = 3
      // axis_for_converter:   [0, 1, 2, 4, 5, 3]
      // axis_for_deconverter: [0, 1, 2, 5, 3, 4]
      int ndim = inputs[0]->ndim();
      std::vector<int> axis_for_converter, axis_for_deconverter;
      for (int i = 0; i < this->axis; i++) {
        axis_for_converter.push_back(i);
        axis_for_deconverter.push_back(i);
      }
      axis_for_deconverter.push_back(ndim - 1);
      for (int i = this->axis + 1; i < ndim; i++) {
        axis_for_converter.push_back(i);
        axis_for_deconverter.push_back(i - 1);
      }
      axis_for_converter.push_back(this->axis);

      // Set up transpose functions.
      Variable dummy_in(inputs[0]->shape()), dummy_mid, dummy_out;
      transpose_converter_ = create_Transpose(this->ctx_, axis_for_converter);
      transpose_deconverter_ =
          create_Transpose(this->ctx_, axis_for_deconverter);
      transpose_converter_->setup({&dummy_in}, {&dummy_mid});
      transpose_deconverter_->setup({&dummy_mid}, {&dummy_out});
      transposed_shape_ = dummy_mid.shape();
    }

    // Calculate size of temporal buffer needed for cub sort.
    // When the first argument of SortPairs(Descending) is NULL, only size
    // calculation of temporal buffer is performed.
    cub_num_items_ = this->total_size;
    cub_num_segments_ = cub_num_items_ / inputs[0]->shape()[this->axis];

    int ofst = 0;
    if (this->reverse) {
      cub::DeviceSegmentedRadixSort::SortPairsDescending<Tcu, int, int *>(
          nullptr, cub_temp_storage_bytes_, nullptr, nullptr, nullptr, nullptr,
          cub_num_items_, cub_num_segments_, &ofst, &ofst);
    } else {
      cub::DeviceSegmentedRadixSort::SortPairs<Tcu, int, int *>(
          nullptr, cub_temp_storage_bytes_, nullptr, nullptr, nullptr, nullptr,
          cub_num_items_, cub_num_segments_, &ofst, &ofst);
    }

    // Index buffer needs same size as input since cub sort algorithm handles
    // input all at once.
    this->temp_index.reshape(inputs[0]->shape(), true);
  }
}

template <typename T>
void SortCuda<T>::thrust_sort(const Variables &inputs,
                              const Variables &outputs) {
#if THRUST_VERSION < 100904 || !defined(THRUST_CPP11) ||                       \
    !defined(THRUST_MODERN_GCC)
  using thrust::sort;
#else
  using thrust::async::sort;
#endif

  using namespace sort_impl;

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

  while (outer_x_raw < x_data + this->total_size) {
    auto inner_x_raw = outer_x_raw;
    auto inner_i_raw = outer_i_raw;

    while (inner_x_raw < outer_x_raw + this->inner_size) {
      auto size = temp_index_var.size();
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(make_sequence, size, temp_index_raw);
      if (this->reverse) {
        (void)sort(thrust::cuda::par.on(0), temp_index_ptr,
                   temp_index_ptr + size,
                   CompareGreater<Tcu>(inner_x_raw, stride));
      } else {
        (void)sort(thrust::cuda::par.on(0), temp_index_ptr,
                   temp_index_ptr + size,
                   CompareLess<Tcu>(inner_x_raw, stride));
      }
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_index, shape[this->axis], stride,
                                     temp_index_raw, inner_i_raw);
      inner_x_raw++;
      inner_i_raw++;
    }
    outer_x_raw += this->outer_size;
    outer_i_raw += this->outer_size;
  }

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
}

template <typename T>
void SortCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  using namespace sort_impl;
  cuda_set_device(this->device_);
  const auto &ctx = this->ctx_;

  if (use_cub_) {

    // Make input memory layout contiguous.
    Variable transposed_input(transposed_shape_);
    if (need_transpose_) {
      transpose_converter_->forward({inputs[0]}, {&transposed_input});
    }

    // Temporary buffer for cub sort algorithm.
    Variable temp_storage(
        Shape_t{static_cast<Size_t>(cub_temp_storage_bytes_)});

    // Make index sequence
    int *temp_index_ptr =
        this->temp_index.template cast_data_and_get_pointer<int>(ctx, true);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(make_index, cub_num_items_,
                                   cub_num_items_ / cub_num_segments_,
                                   temp_index_ptr);

    // Make segment offsets
    Variable d_offsets(Shape_t{static_cast<Size_t>(cub_num_segments_ + 1)});
    int *d_offsets_raw = d_offsets.cast_data_and_get_pointer<int>(ctx, true);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(make_offsets, cub_num_segments_ + 1,
                                   cub_num_items_ / cub_num_segments_,
                                   d_offsets_raw);

    // Device buffers
    char *d_temp_storage = temp_storage.cast_data_and_get_pointer<char>(ctx);

    Variable *v_data_in = need_transpose_ ? &transposed_input : inputs[0];
    const auto *data_in = v_data_in->get_data_pointer<Tcu>(ctx);
    Variable transposed_output(transposed_shape_);
    // When `only_index == true`, `outputs[0]` is used as temporal dummy buffer
    // for sort output.
    Variable *v_data_out = need_transpose_ ? &transposed_output : outputs[0];
    auto *data_out = v_data_out->cast_data_and_get_pointer<Tcu>(ctx, true);

    Variable transposed_index(transposed_shape_);
    Variable *v_index_out =
        need_transpose_ ? &transposed_index : &this->sort_index;
    int *sort_index_ptr =
        v_index_out->template cast_data_and_get_pointer<int>(ctx, true);

    // Call cub sort API
    if (this->reverse) {
      cub::DeviceSegmentedRadixSort::SortPairsDescending(
          d_temp_storage, cub_temp_storage_bytes_, data_in, data_out,
          temp_index_ptr, sort_index_ptr, cub_num_items_, cub_num_segments_,
          d_offsets_raw, d_offsets_raw + 1);
    } else {
      cub::DeviceSegmentedRadixSort::SortPairs(
          d_temp_storage, cub_temp_storage_bytes_, data_in, data_out,
          temp_index_ptr, sort_index_ptr, cub_num_items_, cub_num_segments_,
          d_offsets_raw, d_offsets_raw + 1);
    }

    // Restore memory layout.
    if (need_transpose_) {
      if (!this->only_index) {
        transpose_deconverter_->forward({&transposed_output}, {outputs[0]});
      }
      Variable &sort_index = this->sort_index;
      transpose_deconverter_->forward({&transposed_index}, {&sort_index});
    }
  } else {
    // Thrust sort implementation is used for 64 bits indexing cases.
    thrust_sort(inputs, outputs);
  }

  // Copy sorted index to function output.
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
