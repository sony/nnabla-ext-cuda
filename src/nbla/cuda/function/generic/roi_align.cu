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
#include <nbla/cuda/function/roi_align.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

namespace {
template <typename T> struct Box { T batch_index, x1, y1, x2, y2; };

template <typename T>
__forceinline__ __device__ T clamp(const T x, const T low, const T high) {
  return max(low, min(high, x));
}

template <typename T>
__forceinline__ __device__ int sampling_grid(const int sampling_ratio,
                                             const T step_size) {
  return sampling_ratio > 0 ? sampling_ratio
                            : static_cast<int>(ceil(step_size));
}
}

template <typename T, typename SIZE_T>
__global__ void roi_align_forward_kernel_nchw(
    const SIZE_T size, const T *input_data, const T *boxes_data, T *output_data,
    const SIZE_T samples, const SIZE_T channels, const SIZE_T input_rows,
    const SIZE_T input_cols, const SIZE_T input_stride_c,
    const SIZE_T input_stride_n, const SIZE_T output_rows,
    const SIZE_T output_cols, const SIZE_T output_stride_c,
    const SIZE_T output_stride_n, const int sampling_ratio,
    const float spatial_scale_y, const float spatial_scale_x) {
  NBLA_CUDA_KERNEL_LOOP(output_index, size) {
    SIZE_T n = output_index / output_stride_n;
    SIZE_T index = output_index - n * output_stride_n;
    SIZE_T c = index / output_stride_c;
    index -= c * output_stride_c;
    SIZE_T y = index / output_cols;
    SIZE_T x = index - y * output_cols;

    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    auto const roi_x1 = static_cast<T>(roi.x1 * spatial_scale_x - 0.5f);
    auto const roi_y1 = static_cast<T>(roi.y1 * spatial_scale_y - 0.5f);
    auto const roi_x2 = static_cast<T>(roi.x2 * spatial_scale_x - 0.5f);
    auto const roi_y2 = static_cast<T>(roi.y2 * spatial_scale_y - 0.5f);
    auto const roi_index = clamp<SIZE_T>(roi.batch_index, 0, samples - 1);

    auto const step_size_x = (roi_x2 - roi_x1) / static_cast<T>(output_cols);
    auto const step_size_y = (roi_y2 - roi_y1) / static_cast<T>(output_rows);

    auto const grid_size_x = sampling_grid(sampling_ratio, step_size_x);
    auto const grid_size_y = sampling_grid(sampling_ratio, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const grid_size_xy = grid_size_x * grid_size_y;

    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const xf = roi_x1 + static_cast<T>(x) * step_size_x + half_step_xx;
    auto const yf = roi_y1 + static_cast<T>(y) * step_size_y + half_step_yy;

    auto input_sample_data = input_data + roi_index * input_stride_n;
    auto input_channel_data = input_sample_data + c * input_stride_c;
    auto output_data_value = T(0);

    for (auto yy = 0; yy < grid_size_y; yy++) {
      auto yyf = yf + static_cast<T>(yy) * step_size_yy;

      if (yyf < T(-1) || yyf > static_cast<T>(input_rows))
        continue;

      yyf = clamp<T>(yyf, 0, input_rows - 1);
      auto const y_lo = static_cast<SIZE_T>(yyf);
      auto const y_hi = min(y_lo + 1, input_rows - 1);
      auto const ly = yyf - floor(yyf);
      auto const hy = T(1) - ly;

      for (auto xx = 0; xx < grid_size_x; xx++) {
        auto xxf = xf + static_cast<T>(xx) * step_size_xx;

        if (xxf < T(-1) || xxf > static_cast<T>(input_cols))
          continue;

        xxf = clamp<T>(xxf, 0, input_cols - 1);
        auto const x_lo = static_cast<SIZE_T>(xxf);
        auto const x_hi = min(x_lo + 1, input_cols - 1);
        auto const lx = xxf - floor(xxf);
        auto const hx = T(1) - lx;

        auto const p1 = y_lo * input_cols + x_lo;
        auto const p2 = y_lo * input_cols + x_hi;
        auto const p3 = y_hi * input_cols + x_lo;
        auto const p4 = y_hi * input_cols + x_hi;
        output_data_value += hy * hx * input_channel_data[p1];
        output_data_value += hy * lx * input_channel_data[p2];
        output_data_value += ly * hx * input_channel_data[p3];
        output_data_value += ly * lx * input_channel_data[p4];
      }
    }
    output_data[output_index] = output_data_value / grid_size_xy;
  }
}

template <typename T, typename SIZE_T>
__global__ void roi_align_forward_kernel_nhwc(
    const SIZE_T size, const T *input_data, const T *boxes_data, T *output_data,
    const SIZE_T samples, const SIZE_T channels, const SIZE_T input_rows,
    const SIZE_T input_cols, const SIZE_T input_stride_n,
    const SIZE_T output_rows, const SIZE_T output_cols,
    const SIZE_T output_size_xy, const int sampling_ratio,
    const float spatial_scale_y, const float spatial_scale_x) {
  NBLA_CUDA_KERNEL_LOOP(thread_index, size) {
    SIZE_T n = thread_index / output_size_xy;
    SIZE_T i = thread_index - n * output_size_xy;
    SIZE_T y = i / output_cols;
    SIZE_T x = i - y * output_cols;

    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    auto const roi_x1 = static_cast<T>(roi.x1 * spatial_scale_x - 0.5f);
    auto const roi_y1 = static_cast<T>(roi.y1 * spatial_scale_y - 0.5f);
    auto const roi_x2 = static_cast<T>(roi.x2 * spatial_scale_x - 0.5f);
    auto const roi_y2 = static_cast<T>(roi.y2 * spatial_scale_y - 0.5f);
    auto const roi_index = clamp<SIZE_T>(roi.batch_index, 0, samples - 1);

    auto const step_size_x = (roi_x2 - roi_x1) / static_cast<T>(output_cols);
    auto const step_size_y = (roi_y2 - roi_y1) / static_cast<T>(output_rows);

    auto const grid_size_x = sampling_grid(sampling_ratio, step_size_x);
    auto const grid_size_y = sampling_grid(sampling_ratio, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const grid_size_xy = grid_size_x * grid_size_y;

    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const xf = roi_x1 + static_cast<T>(x) * step_size_x + half_step_xx;
    auto const yf = roi_y1 + static_cast<T>(y) * step_size_y + half_step_yy;

    auto input_sample_data = input_data + roi_index * input_stride_n;
    auto output_channel_data = output_data + thread_index * channels;

    for (auto c = 0; c < channels; c++) {
      output_channel_data[c] = T(0);
    }

    for (auto yy = 0; yy < grid_size_y; yy++) {
      auto yyf = yf + static_cast<T>(yy) * step_size_yy;

      if (yyf < T(-1) || yyf > static_cast<T>(input_rows))
        continue;

      yyf = clamp<T>(yyf, 0, input_rows - 1);
      auto const y_lo = static_cast<SIZE_T>(yyf);
      auto const y_hi = min(y_lo + 1, input_rows - 1);
      auto const ly = yyf - floor(yyf);
      auto const hy = T(1) - ly;

      for (auto xx = 0; xx < grid_size_x; xx++) {
        auto xxf = xf + static_cast<T>(xx) * step_size_xx;

        if (xxf < T(-1) || xxf > static_cast<T>(input_cols))
          continue;

        xxf = clamp<T>(xxf, 0, input_cols - 1);
        auto const x_lo = static_cast<SIZE_T>(xxf);
        auto const x_hi = min(x_lo + 1, input_cols - 1);
        auto const lx = xxf - floor(xxf);
        auto const hx = T(1) - lx;

        auto const p1 = (y_lo * input_cols + x_lo) * channels;
        auto const p2 = (y_lo * input_cols + x_hi) * channels;
        auto const p3 = (y_hi * input_cols + x_lo) * channels;
        auto const p4 = (y_hi * input_cols + x_hi) * channels;
        for (auto c = 0; c < channels; c++) {
          auto output_data_value = T(0);
          output_data_value += hy * hx * input_sample_data[p1 + c];
          output_data_value += hy * lx * input_sample_data[p2 + c];
          output_data_value += ly * hx * input_sample_data[p3 + c];
          output_data_value += ly * lx * input_sample_data[p4 + c];
          output_channel_data[c] += output_data_value;
        }
      }
    }
    for (auto c = 0; c < channels; c++) {
      output_channel_data[c] /= grid_size_xy;
    }
  }
}

template <typename T, typename SIZE_T>
__global__ void roi_align_backward_kernel_nchw(
    const SIZE_T size, T *input_grad, const T *boxes_data, const T *output_grad,
    const SIZE_T samples, const SIZE_T channels, const SIZE_T input_rows,
    const SIZE_T input_cols, const SIZE_T input_channel_size,
    const SIZE_T input_sample_size, const SIZE_T output_rows,
    const SIZE_T output_cols, const SIZE_T output_stride_c,
    const SIZE_T output_stride_n, const int sampling_ratio,
    const float spatial_scale_y, const float spatial_scale_x) {
  NBLA_CUDA_KERNEL_LOOP(output_index, size) {
    SIZE_T n = output_index / output_stride_n;
    SIZE_T index = output_index - n * output_stride_n;
    SIZE_T c = index / output_stride_c;
    index -= c * output_stride_c;
    SIZE_T y = index / output_cols;
    SIZE_T x = index - y * output_cols;

    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    auto const roi_x1 = static_cast<T>(roi.x1 * spatial_scale_x - 0.5f);
    auto const roi_y1 = static_cast<T>(roi.y1 * spatial_scale_y - 0.5f);
    auto const roi_x2 = static_cast<T>(roi.x2 * spatial_scale_x - 0.5f);
    auto const roi_y2 = static_cast<T>(roi.y2 * spatial_scale_y - 0.5f);
    auto const roi_index = clamp<SIZE_T>(roi.batch_index, 0, samples - 1);

    auto const step_size_x = (roi_x2 - roi_x1) / static_cast<T>(output_cols);
    auto const step_size_y = (roi_y2 - roi_y1) / static_cast<T>(output_rows);

    auto const grid_size_x = sampling_grid(sampling_ratio, step_size_x);
    auto const grid_size_y = sampling_grid(sampling_ratio, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const grid_size_xy = grid_size_x * grid_size_y;

    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const xf = roi_x1 + static_cast<T>(x) * step_size_x + half_step_xx;
    auto const yf = roi_y1 + static_cast<T>(y) * step_size_y + half_step_yy;

    auto input_sample_grad = input_grad + roi_index * input_sample_size;
    auto input_channel_grad = input_sample_grad + c * input_channel_size;
    auto output_grad_value = output_grad[output_index] / grid_size_xy;

    for (auto yy = 0; yy < grid_size_y; yy++) {
      auto yyf = yf + static_cast<T>(yy) * step_size_yy;

      if (yyf < T(-1) || yyf > static_cast<T>(input_rows))
        continue;

      yyf = clamp<T>(yyf, 0, input_rows - 1);
      auto const y_lo = static_cast<SIZE_T>(yyf);
      auto const y_hi = min(y_lo + 1, input_rows - 1);
      auto const ly = yyf - floor(yyf);
      auto const hy = T(1) - ly;

      for (auto xx = 0; xx < grid_size_x; xx++) {
        auto xxf = xf + static_cast<T>(xx) * step_size_xx;

        if (xxf < T(-1) || xxf > static_cast<T>(input_cols))
          continue;

        xxf = clamp<T>(xxf, 0, input_cols - 1);
        auto const x_lo = static_cast<SIZE_T>(xxf);
        auto const x_hi = min(x_lo + 1, input_cols - 1);
        auto const lx = xxf - floor(xxf);
        auto const hx = T(1) - lx;

        auto const p1 = y_lo * input_cols + x_lo;
        auto const p2 = y_lo * input_cols + x_hi;
        auto const p3 = y_hi * input_cols + x_lo;
        auto const p4 = y_hi * input_cols + x_hi;
        atomic_add(&input_channel_grad[p1], hy * hx * output_grad_value);
        atomic_add(&input_channel_grad[p2], hy * lx * output_grad_value);
        atomic_add(&input_channel_grad[p3], ly * hx * output_grad_value);
        atomic_add(&input_channel_grad[p4], ly * lx * output_grad_value);
      }
    }
  }
}

template <typename T, typename SIZE_T>
__global__ void roi_align_backward_kernel_nhwc(
    const SIZE_T size, T *input_grad, const T *boxes_data, const T *output_grad,
    const SIZE_T samples, const SIZE_T channels, const SIZE_T input_rows,
    const SIZE_T input_cols, const SIZE_T input_stride_n,
    const SIZE_T output_rows, const SIZE_T output_cols,
    const SIZE_T output_size_xy, const int sampling_ratio,
    const float spatial_scale_y, const float spatial_scale_x) {
  NBLA_CUDA_KERNEL_LOOP(thread_index, size) {
    SIZE_T n = thread_index / output_size_xy;
    SIZE_T i = thread_index - n * output_size_xy;
    SIZE_T y = i / output_cols;
    SIZE_T x = i - y * output_cols;

    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    auto const roi_x1 = static_cast<T>(roi.x1 * spatial_scale_x - 0.5f);
    auto const roi_y1 = static_cast<T>(roi.y1 * spatial_scale_y - 0.5f);
    auto const roi_x2 = static_cast<T>(roi.x2 * spatial_scale_x - 0.5f);
    auto const roi_y2 = static_cast<T>(roi.y2 * spatial_scale_y - 0.5f);
    auto const roi_index = clamp<SIZE_T>(roi.batch_index, 0, samples - 1);

    auto const step_size_x = (roi_x2 - roi_x1) / static_cast<T>(output_cols);
    auto const step_size_y = (roi_y2 - roi_y1) / static_cast<T>(output_rows);

    auto const grid_size_x = sampling_grid(sampling_ratio, step_size_x);
    auto const grid_size_y = sampling_grid(sampling_ratio, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const grid_size_xy = grid_size_x * grid_size_y;

    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const xf = roi_x1 + static_cast<T>(x) * step_size_x + half_step_xx;
    auto const yf = roi_y1 + static_cast<T>(y) * step_size_y + half_step_yy;

    auto input_sample_grad = input_grad + roi_index * input_stride_n;
    auto output_channel_grad = output_grad + thread_index * channels;

    for (auto yy = 0; yy < grid_size_y; yy++) {
      auto yyf = yf + static_cast<T>(yy) * step_size_yy;

      if (yyf < T(-1) || yyf > static_cast<T>(input_rows))
        continue;

      yyf = clamp<T>(yyf, 0, input_rows - 1);
      auto const y_lo = static_cast<SIZE_T>(yyf);
      auto const y_hi = min(y_lo + 1, input_rows - 1);
      auto const ly = yyf - floor(yyf);
      auto const hy = T(1) - ly;

      for (auto xx = 0; xx < grid_size_x; xx++) {
        auto xxf = xf + static_cast<T>(xx) * step_size_xx;

        if (xxf < T(-1) || xxf > static_cast<T>(input_cols))
          continue;

        xxf = clamp<T>(xxf, 0, input_cols - 1);
        auto const x_lo = static_cast<SIZE_T>(xxf);
        auto const x_hi = min(x_lo + 1, input_cols - 1);
        auto const lx = xxf - floor(xxf);
        auto const hx = T(1) - lx;

        auto const p1 = (y_lo * input_cols + x_lo) * channels;
        auto const p2 = (y_lo * input_cols + x_hi) * channels;
        auto const p3 = (y_hi * input_cols + x_lo) * channels;
        auto const p4 = (y_hi * input_cols + x_hi) * channels;
        for (auto c = 0; c < channels; c++) {
          auto const output_grad_value = output_channel_grad[c] / grid_size_xy;
          atomic_add(&input_sample_grad[p1 + c], hy * hx * output_grad_value);
          atomic_add(&input_sample_grad[p2 + c], hy * lx * output_grad_value);
          atomic_add(&input_sample_grad[p3 + c], ly * hx * output_grad_value);
          atomic_add(&input_sample_grad[p4 + c], ly * lx * output_grad_value);
        }
      }
    }
  }
}

template <typename T>
void RoiAlignCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  RoiAlign<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void RoiAlignCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);

  auto input = inputs.at(0);
  auto boxes = inputs.at(1);
  auto output = outputs.at(0);

  auto input_data = input->get_data_pointer<Tcu>(this->ctx_);
  auto boxes_data = boxes->get_data_pointer<Tcu>(this->ctx_);
  auto output_data = output->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto const samples = input->shape().at(0);
  auto const spatial_scale_y = this->spatial_scale_.at(0);
  auto const spatial_scale_x = this->spatial_scale_.at(1);

  if (!this->channel_last_) {
    auto const channels = input->shape().at(1);
    auto const input_rows = input->shape().at(2);
    auto const input_cols = input->shape().at(3);
    auto const output_rows = output->shape().at(2);
    auto const output_cols = output->shape().at(3);
    auto const input_stride_n = input->strides().at(0);
    auto const input_stride_c = input->strides().at(1);
    auto const output_stride_n = output->strides().at(0);
    auto const output_stride_c = output->strides().at(1);
    auto const nthreads = output->size();

    if (output->size() <= INT32_MAX) {
      auto kernel = roi_align_forward_kernel_nchw<Tcu, int32_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_data, boxes_data, output_data, samples,
          channels, input_rows, input_cols, input_stride_c, input_stride_n,
          output_rows, output_cols, output_stride_c, output_stride_n,
          this->sampling_ratio_, spatial_scale_y, spatial_scale_x);
    } else {
      auto kernel = roi_align_forward_kernel_nchw<Tcu, int64_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_data, boxes_data, output_data, samples,
          channels, input_rows, input_cols, input_stride_c, input_stride_n,
          output_rows, output_cols, output_stride_c, output_stride_n,
          this->sampling_ratio_, spatial_scale_y, spatial_scale_x);
    }
  } else {
    auto const channels = input->shape().at(3);
    auto const input_rows = input->shape().at(1);
    auto const input_cols = input->shape().at(2);
    auto const output_rows = output->shape().at(1);
    auto const output_cols = output->shape().at(2);
    auto const input_stride_n = input->strides().at(0);
    auto const output_size_xy = output_rows * output_cols;
    auto const nthreads = output->size() / channels;

    if (output->size() <= INT32_MAX) {
      auto kernel = roi_align_forward_kernel_nhwc<Tcu, int32_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_data, boxes_data, output_data, samples,
          channels, input_rows, input_cols, input_stride_n, output_rows,
          output_cols, output_size_xy, this->sampling_ratio_, spatial_scale_y,
          spatial_scale_x);
    } else {
      auto kernel = roi_align_forward_kernel_nhwc<Tcu, int64_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_data, boxes_data, output_data, samples,
          channels, input_rows, input_cols, input_stride_n, output_rows,
          output_cols, output_size_xy, this->sampling_ratio_, spatial_scale_y,
          spatial_scale_x);
    }
  }
}

template <typename T>
void RoiAlignCuda<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {

  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(this->device_);

  auto input = inputs.at(0);
  auto boxes = inputs.at(1);
  auto output = outputs.at(0);

  auto input_grad = input->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
  auto boxes_data = boxes->get_data_pointer<Tcu>(this->ctx_);
  auto output_grad = output->get_grad_pointer<Tcu>(this->ctx_);

  auto const samples = input->shape().at(0);
  auto const spatial_scale_y = this->spatial_scale_.at(0);
  auto const spatial_scale_x = this->spatial_scale_.at(1);

  if (!this->channel_last_) {
    auto const channels = input->shape().at(1);
    auto const input_rows = input->shape().at(2);
    auto const input_cols = input->shape().at(3);
    auto const output_rows = output->shape().at(2);
    auto const output_cols = output->shape().at(3);
    auto const input_stride_n = input->strides().at(0);
    auto const input_stride_c = input->strides().at(1);
    auto const output_stride_n = output->strides().at(0);
    auto const output_stride_c = output->strides().at(1);
    auto const nthreads = output->size();

    if (output->size() <= INT32_MAX) {
      auto kernel = roi_align_backward_kernel_nchw<Tcu, int32_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_grad, boxes_data, output_grad, samples,
          channels, input_rows, input_cols, input_stride_c, input_stride_n,
          output_rows, output_cols, output_stride_c, output_stride_n,
          this->sampling_ratio_, spatial_scale_y, spatial_scale_x);
    } else {
      auto kernel = roi_align_backward_kernel_nchw<Tcu, int64_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_grad, boxes_data, output_grad, samples,
          channels, input_rows, input_cols, input_stride_c, input_stride_n,
          output_rows, output_cols, output_stride_c, output_stride_n,
          this->sampling_ratio_, spatial_scale_y, spatial_scale_x);
    }
  } else {
    auto const channels = input->shape().at(3);
    auto const input_rows = input->shape().at(1);
    auto const input_cols = input->shape().at(2);
    auto const output_rows = output->shape().at(1);
    auto const output_cols = output->shape().at(2);
    auto const input_stride_n = input->strides().at(0);
    auto const output_size_xy = output_rows * output_cols;
    auto const nthreads = output->size() / channels;

    if (output->size() <= INT32_MAX) {
      auto kernel = roi_align_backward_kernel_nhwc<Tcu, int32_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_grad, boxes_data, output_grad, samples,
          channels, input_rows, input_cols, input_stride_n, output_rows,
          output_cols, output_size_xy, this->sampling_ratio_, spatial_scale_y,
          spatial_scale_x);
    } else {
      auto kernel = roi_align_backward_kernel_nhwc<Tcu, int64_t>;

      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, nthreads, input_grad, boxes_data, output_grad, samples,
          channels, input_rows, input_cols, input_stride_n, output_rows,
          output_cols, output_size_xy, this->sampling_ratio_, spatial_scale_y,
          spatial_scale_x);
    }
  }
}
}
