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
#include <nbla/cuda/function/pow2_quantize.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_quantize_forward(const int num, T *y, const T *x,
                                        const bool sign, const bool with_zero,
                                        const float p_max, const float p_min,
                                        const float pruning_threshold) {
  T q;
  T x_idx;
  T x_abs;
  bool sign_x;
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    // quantize in positive domain
    x_idx = x[idx];
    sign_x = (x_idx < 0.0);
    x_abs = fabs(x_idx);
    q = powf(2., round(log2f(x_abs)));
    if (q > p_max) {
      q = p_max;
    } else if (q < p_min && with_zero) {
      q = x_abs < pruning_threshold ? (T)0. : (T)p_min;
    } else if (q < p_min) {
      q = p_min;
    }

    // address sign
    sign_x = (x_idx < 0.);
    if (sign) {
      q = sign_x ? -q : q;
    } else {
      if (with_zero) {
        q = sign_x ? (T)0. : q;
      } else {
        q = sign_x ? (T)p_min : q;
      }
    }
    y[idx] = q;
  }
}

template <typename T>
void Pow2QuantizeCuda<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // TODO: consider arithmetic mean
  cuda_set_device(std::stoi(this->ctx_.device_id));

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);
  size_t size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_quantize_forward, size, y, x,
                                 this->sign_, this->with_zero_, this->p_max_,
                                 this->p_min_, this->pruning_threshold_);
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
                                         const T *x, const bool sign,
                                         const bool with_zero,
                                         const float p_max, const float p_min,
                                         const float pruning_threshold) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T q;
    T c;
    T x_abs = fabs(x[idx]);
    q = powf(2., round(log2f(x_abs)));
    c = 1.; // normally, assume grad is 1
    if (q > p_max) {
      c = 0.;
    }

    // address sign
    if (!sign) {
      bool sign_x;
      sign_x = (x[idx] < 0.0);
      c = sign_x ? (T)0. : c;
    }

    if (accum) {
      dx[idx] += c * dy[idx];
    } else {
      dx[idx] = c * dy[idx];
    }
  }
}

template <typename T>
void Pow2QuantizeCuda<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  // TODO: consider fine-grained STE
  cuda_set_device(std::stoi(this->ctx_.device_id));

  if (!propagate_down[0]) {
    return;
  }

  Size_t size = inputs[0]->size();
  const Tc *x = inputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);

  if (this->ste_fine_grained_) {
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_quantize_backward<Tc, true>), size,
                                     dx, dy, x, this->sign_, this->with_zero_,
                                     this->p_max_, this->p_min_,
                                     this->pruning_threshold_);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_quantize_backward<Tc, false>),
                                     size, dx, dy, x, this->sign_,
                                     this->with_zero_, this->p_max_,
                                     this->p_min_, this->pruning_threshold_);
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
