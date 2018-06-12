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
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
int get_shape_from_last(const vector<T> &shape, int index,
                        int default_value = 1) {
  int axis = shape.size() - 1 + index;
  if (axis < 0) {
    return default_value;
  }
  return shape[axis];
}

template <typename T>
void PadCuda<T>::setup_impl(const Variables &inputs, const Variables &outputs) {

  Pad<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
__global__ void
kernel_pad_nD_forward(const int num, const T *x, T *y, const float value,
                      int in_size, int p_front, int p_top, int p_left,
                      int i_channel, int i_height, int i_width, int o_channel,
                      int o_height, int o_width) {
  NBLA_CUDA_KERNEL_LOOP(i, num) {
    int b = i / (o_channel * o_height * o_width);
    int c = (i / (o_height * o_width)) % o_channel;
    int h = (i / o_width) % o_height;
    int w = i % o_width;

    // Calculate input index
    int ib = b;
    int ic = c - p_front;
    int ih = h - p_top;
    int iw = w - p_left;
    int j = ((ib * i_channel + ic) * i_height + ih) * i_width + iw;

    if ((c >= p_front && c < (p_front + i_channel)) &&
        (w >= p_left && w < (p_left + i_width)) &&
        (h >= p_top && h < (p_top + i_height))) {
      if (j <= in_size) {
        y[i] = x[j];
      }
    } else {
      y[i] = (T)value;
    }
  }
}

template <typename T>
void PadCuda<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  // If output NDarray is not the size of 4D, then convert to 4D by adding dummy
  // dimensions (i.e. 1) or
  // if dimension is more than 4D then squeeze first dimensions by
  // multiplication except last 3 dimensions.
  int o_channel = get_shape_from_last(outputs[0]->shape(), -2);
  int o_height = get_shape_from_last(outputs[0]->shape(), -1);
  int o_width = get_shape_from_last(outputs[0]->shape(), 0);
  int o_batch = outputs[0]->size() / (o_channel * o_height * o_width);

  int i_channel = get_shape_from_last(inputs[0]->shape(), -2);
  int i_height = get_shape_from_last(inputs[0]->shape(), -1);
  int i_width = get_shape_from_last(inputs[0]->shape(), 0);
  int i_batch = inputs[0]->size() / (i_channel * i_height * i_width);

  NBLA_CHECK(
      i_batch == o_batch, error_code::value,
      " Internal error: Input array and output array batch size not same.");

  // If pad_width_ not of size 3D, then convert to 3D by adding dummy padding
  // (i.e. 0).
  int p_front = get_shape_from_last(this->pad_width_, -5, 0);
  int p_top = get_shape_from_last(this->pad_width_, -3, 0);
  int p_left = get_shape_from_last(this->pad_width_, -1, 0);

  switch (this->pad_mode_[this->mode_]) {
  case Pad<T>::p_constant:
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_pad_nD_forward, outputs[0]->size(), x,
                                   y, this->constant_value_, inputs[0]->size(),
                                   p_front, p_top, p_left, i_channel, i_height,
                                   i_width, o_channel, o_height, o_width);
    break;
  case Pad<T>::p_replicate: // TODO
  case Pad<T>::p_reflect:   // TODO
  default:
    NBLA_CHECK(false, error_code::value,
               " Internal error: pad mode is not supported.");
    break;
  }
}

template <typename T, bool accum>
__global__ void kernel_pad_nD_backward(const int num, T *dx, const T *dy,
                                       int in_size, int p_front, int p_top,
                                       int p_left, int i_channel, int i_height,
                                       int i_width, int o_channel, int o_height,
                                       int o_width) {
  NBLA_CUDA_KERNEL_LOOP(i, num) {
    int b = i / (o_channel * o_height * o_width);
    int c = (i / (o_height * o_width)) % o_channel;
    int h = (i / o_width) % o_height;
    int w = i % o_width;

    // Calculate input index
    int ib = b;
    int ic = c - p_front;
    int ih = h - p_top;
    int iw = w - p_left;
    int j = ((ib * i_channel + ic) * i_height + ih) * i_width + iw;

    if ((c >= p_front && c < (p_front + i_channel)) &&
        (w >= p_left && w < (p_left + i_width)) &&
        (h >= p_top && h < (p_top + i_height))) {
      if (j <= in_size) {
        if (accum) {
          dx[j] += dy[i];
        } else {
          dx[j] = dy[i];
        }
      }
    }
  }
}

template <typename T>
void PadCuda<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  Tcu *dx{nullptr};
  dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  // If output NDarray is not the size of 4D, then convert to 4D by adding dummy
  // dimensions (i.e. 1) or
  // if dimension is more than 4D then squeeze first dimensions by
  // multiplication except last 3 dimensions.
  int o_channel = get_shape_from_last(outputs[0]->shape(), -2);
  int o_height = get_shape_from_last(outputs[0]->shape(), -1);
  int o_width = get_shape_from_last(outputs[0]->shape(), 0);
  int o_batch = outputs[0]->size() / (o_channel * o_height * o_width);

  int i_channel = get_shape_from_last(inputs[0]->shape(), -2);
  int i_height = get_shape_from_last(inputs[0]->shape(), -1);
  int i_width = get_shape_from_last(inputs[0]->shape(), 0);
  int i_batch = inputs[0]->size() / (i_channel * i_height * i_width);

  NBLA_CHECK(
      i_batch == o_batch, error_code::value,
      " Internal error: Input array and output array batch size not same.");

  // If pad_width_ not of size 3D, then convert to 3D by adding dummy padding
  // (i.e. 0).
  int p_front = get_shape_from_last(this->pad_width_, -5, 0);
  int p_top = get_shape_from_last(this->pad_width_, -3, 0);
  int p_left = get_shape_from_last(this->pad_width_, -1, 0);

  switch (this->pad_mode_[this->mode_]) {
  case Pad<T>::p_constant:
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_pad_nD_backward<Tcu, true>), outputs[0]->size(), dx, dy,
          inputs[0]->size(), p_front, p_top, p_left, i_channel, i_height,
          i_width, o_channel, o_height, o_width);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_pad_nD_backward<Tcu, false>), outputs[0]->size(), dx, dy,
          inputs[0]->size(), p_front, p_top, p_left, i_channel, i_height,
          i_width, o_channel, o_height, o_width);
    }
    break;
  case Pad<T>::p_replicate: // TODO
  case Pad<T>::p_reflect:   // TODO
  default:
    NBLA_CHECK(false, error_code::value,
               " Internal error: pad mode is not supported.");
    break;
  }
}
}
