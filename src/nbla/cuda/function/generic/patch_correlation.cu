// Copyright 2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/function/patch_correlation.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

// TODO: Remove these #includes. Only for debug.
#include <iostream>
#include <typeinfo>

namespace nbla {

typedef struct InputShapeAndStride {
  int h;   // height
  int w;   // width
  int c;   // channels
  int wc;  // w * c
  int hwc; // h * w * c
} InputShapeAndStride;

namespace patch_correlation {

template <typename T>
__global__ void forward(const int size, const InputShapeAndStride input,
                        const int4 ostride, const int2 patch, const int2 shift,
                        const int2 patch_step, const int2 shift_step,
                        const int4 padding, const T *in1_data,
                        const T *in2_data, T *out_data) {
  // Input data is 4 dims NHWC. Output data is 5 dims NHWCyCx.
  // ostride.(w|z|y|x) = (H*W*CY*CX|W*CY*CX|CY*CX|CX)
  // padding.(w|z|y|x) = (top|bottom|left|right)
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    auto out_index = index;
    auto n = index / ostride.w;
    index -= n * ostride.w;
    auto y = index / ostride.z;
    index -= y * ostride.z;
    auto x = index / ostride.y;
    index -= x * ostride.y;
    auto cy = index / ostride.x;
    auto cx = index - cy * ostride.x;

    auto ya = y * patch_step.y - padding.w;
    auto xa = x * patch_step.x - padding.y;
    auto yb = ya - shift.y + cy * shift_step.y;
    auto xb = xa - shift.x + cx * shift_step.x;

    auto value = T{0};

    for (int ky = 0; ky < patch.y; ky++) {
      auto iya = ya + ky;
      auto iyb = yb + ky;
      if ((0 <= iya) && (iya < input.h) && (0 <= iyb) && (iyb < input.h)) {
        for (int kx = 0; kx < patch.x; kx++) {
          auto ixa = xa + kx;
          auto ixb = xb + kx;
          if ((0 <= ixa) && (ixa < input.w) && (0 <= ixb) && (ixb < input.w)) {
            auto in1_index = n * input.hwc + iya * input.wc + ixa * input.c;
            auto in2_index = n * input.hwc + iyb * input.wc + ixb * input.c;
            for (int c = 0; c < input.c; c++) {
              value += in1_data[in1_index + c] * in2_data[in2_index + c];
            }
          }
        }
      }
    }
    out_data[out_index] = value;
  }
}

template <typename T, bool propagate_down_in1, bool propagate_down_in2>
__global__ void backward(const int size, const InputShapeAndStride input,
                         const int4 ostride, const int2 patch, const int2 shift,
                         const int2 patch_step, const int2 shift_step,
                         const int4 padding, const T *out_g, const T *in1_d,
                         const T *in2_d, T *__restrict__ in1_g,
                         T *__restrict__ in2_g) {
  // Input data is 4 dims NHWC. Output data is 5 dims NHWCyCx.
  // ostride.(w|z|y|x) = (H*W*CY*CX|W*CY*CX|CY*CX|CX)
  // padding.(w|z|y|x) = (top|bottom|left|right)
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    auto out_index = index;
    auto n = index / ostride.w;
    index -= n * ostride.w;
    auto y = index / ostride.z;
    index -= y * ostride.z;
    auto x = index / ostride.y;
    index -= x * ostride.y;
    auto cy = index / ostride.x;
    auto cx = index - cy * ostride.x;

    auto ya = y * patch_step.y - padding.w;
    auto xa = x * patch_step.x - padding.y;
    auto yb = ya - shift.y + cy * shift_step.y;
    auto xb = xa - shift.x + cx * shift_step.x;

    auto grad = out_g[out_index];

    for (int ky = 0; ky < patch.y; ky++) {
      auto iya = ya + ky;
      auto iyb = yb + ky;
      if ((0 <= iya) && (iya < input.h) && (0 <= iyb) && (iyb < input.h)) {
        for (int kx = 0; kx < patch.x; kx++) {
          auto ixa = xa + kx;
          auto ixb = xb + kx;
          if ((0 <= ixa) && (ixa < input.w) && (0 <= ixb) && (ixb < input.w)) {
            auto in1_index = n * input.hwc + iya * input.wc + ixa * input.c;
            auto in2_index = n * input.hwc + iyb * input.wc + ixb * input.c;
            for (int c = 0; c < input.c; c++) {
              if (propagate_down_in1) {
                auto g = in2_d[in2_index + c] * grad;
                atomic_add(&in1_g[in1_index + c], g);
              }
              if (propagate_down_in2) {
                auto g = in1_d[in1_index + c] * grad;
                atomic_add(&in2_g[in2_index + c], g);
              }
            }
          }
        }
      }
    }
  }
}
}

template <typename T>
void PatchCorrelationCuda<T>::setup_impl(const Variables &inputs,
                                         const Variables &outputs) {
  PatchCorrelation<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void PatchCorrelationCuda<T>::forward_impl(const Variables &inputs,
                                           const Variables &outputs) {
  cuda_set_device(this->device_);
  auto in1_data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto in2_data = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto out_data = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto const patch = make_int2(this->patch_[1], this->patch_[0]);
  auto const shift = make_int2(this->shift_[1], this->shift_[0]);
  auto const patch_step = make_int2(this->patch_step_[1], this->patch_step_[0]);
  auto const shift_step = make_int2(this->shift_step_[1], this->shift_step_[0]);
  auto const padding = make_int4(this->padding_[3], this->padding_[2],
                                 this->padding_[1], this->padding_[0]);

  auto ostride = make_int4(outputs[0]->strides()[3], outputs[0]->strides()[2],
                           outputs[0]->strides()[1], outputs[0]->strides()[0]);

  InputShapeAndStride input;
  input.h = static_cast<int>(inputs[0]->shape()[1]);
  input.w = static_cast<int>(inputs[0]->shape()[2]);
  input.c = static_cast<int>(inputs[0]->shape()[3]);
  input.wc = input.w * input.c;
  input.hwc = input.h * input.wc;

  using patch_correlation::forward;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward, outputs[0]->size(), input, ostride,
                                 patch, shift, patch_step, shift_step, padding,
                                 in1_data, in2_data, out_data);
}

template <typename T>
void PatchCorrelationCuda<T>::backward_impl(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  auto const patch = make_int2(this->patch_[1], this->patch_[0]);
  auto const shift = make_int2(this->shift_[1], this->shift_[0]);
  auto const patch_step = make_int2(this->patch_step_[1], this->patch_step_[0]);
  auto const shift_step = make_int2(this->shift_step_[1], this->shift_step_[0]);
  auto const padding = make_int4(this->padding_[3], this->padding_[2],
                                 this->padding_[1], this->padding_[0]);

  auto ostride = make_int4(outputs[0]->strides()[3], outputs[0]->strides()[2],
                           outputs[0]->strides()[1], outputs[0]->strides()[0]);

  InputShapeAndStride input;
  input.h = static_cast<int>(inputs[0]->shape()[1]);
  input.w = static_cast<int>(inputs[0]->shape()[2]);
  input.c = static_cast<int>(inputs[0]->shape()[3]);
  input.wc = input.w * input.c;
  input.hwc = input.h * input.wc;

  auto out_g = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  if (propagate_down[0] && propagate_down[1]) {
    auto in1_d = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
    auto in2_d = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
    auto in1_g = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto in2_g = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto backward = patch_correlation::backward<Tcu, true, true>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward, outputs[0]->size(), input, ostride,
                                   patch, shift, patch_step, shift_step,
                                   padding, out_g, in1_d, in2_d, in1_g, in2_g);
  } else if (propagate_down[0]) {
    auto in1_g = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto in2_d = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
    auto backward = patch_correlation::backward<Tcu, true, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward, outputs[0]->size(), input, ostride, patch, shift, patch_step,
        shift_step, padding, out_g, nullptr, in2_d, in1_g, nullptr);
  } else if (propagate_down[1]) {
    auto in2_g = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto in1_d = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
    auto backward = patch_correlation::backward<Tcu, false, true>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward, outputs[0]->size(), input, ostride, patch, shift, patch_step,
        shift_step, padding, out_g, in1_d, nullptr, nullptr, in2_g);
  }
}
}
