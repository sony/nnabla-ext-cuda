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

/** Quantize
 */
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/fixed_point_quantize.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_quantize_forward(const int num, T *y, const T *x,
                                        const float max, const float min,
                                        const float delta) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T x_idx = x[idx];
    bool sign_x;
    T abs_x;
    T y_tmp;

    if (x[idx] > max) {
      y_tmp = max;
    } else if (x_idx < min) {
      y_tmp = min;
    } else {
      sign_x = (x_idx < 0.);
      abs_x = fabs(x_idx);
      y_tmp = floor(abs_x / delta + 0.5) * delta;
      y_tmp = sign_x ? -y_tmp : y_tmp;
    }
    y[idx] = y_tmp;
  }
}

template <typename T>
void FixedPointQuantizeCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_quantize_forward, size, y, x,
                                 this->max_, this->min_, this->delta_);
}

template <typename T, bool accum = true>
__global__ void kernel_naive_quantize_backward(const int num, T *dx,
                                               const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    if (accum) {
      dx[idx] += dy[idx];
    } else {
      dx[idx] = dy[idx];
    }
  }
}

template <typename T, bool accum = true>
__global__ void kernel_quantize_backward(const int num, T *dx, const T *dy,
                                         const T *x, const float max,
                                         const float min) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    if (x[idx] > max) {
      if (!accum)
        dx[idx] = (T)0.;
    } else if (x[idx] < min) { // also consider sign or unsign.
      if (!accum)
        dx[idx] = (T)0.;
    } else { // non-clipped region
      if (accum) {
        dx[idx] += dy[idx];
      } else {
        dx[idx] = dy[idx];
      }
    }
  }
}

template <typename T>
void FixedPointQuantizeCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  // TODO: consider fine-grained STE
  cuda_set_device(std::stoi(this->ctx_.device_id));

  if (!propagate_down[0]) {
    return;
  }

  Size_t size = inputs[0]->size();
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  if (this->ste_fine_grained_) {
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_quantize_backward<Tc, true>), size,
                                     dx, dy, x, this->max_, this->min_);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_quantize_backward<Tc, false>),
                                     size, dx, dy, x, this->max_, this->min_);
    }
  } else {
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_naive_quantize_backward<Tc, true>),
                                     size, dx, dy);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_naive_quantize_backward<Tc, false>), size, dx, dy);
    }
  }
}
} // namespace nbla
