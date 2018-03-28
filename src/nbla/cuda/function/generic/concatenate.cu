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

// concatenate.cpp

#include <nbla/array.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/concatenate.hpp>

namespace nbla {

// size: inner size x outer size
template <typename T>
__global__ void
kernel_concatenate_forward(const int size, const int inner_total_size,
                           const int inner_size, const int inner_offset,
                           const T *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    const int o = index / inner_size;
    const int c = index % inner_size;
    y[o * inner_total_size + inner_offset + c] = x[index];
  }
}

template <typename T, bool accum>
__global__ void
kernel_concatenate_backward(const int size, const int inner_total_size,
                            const int inner_size, const int inner_offset,
                            const T *dy, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    const int o = index / inner_size;
    const int c = index % inner_size;
    dx[index] = (accum ? dx[index] : (T)0) +
                dy[o * inner_total_size + inner_offset + c];
  }
}

template <typename T>
void ConcatenateCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  Concatenate<T>::setup_impl(inputs, outputs);
}

template <class T>
void ConcatenateCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  int inner_offset = 0;
  for (int c = 0; c < inputs.size(); ++c) {
    const Tc *x = inputs[c]->get_data_pointer<Tc>(this->ctx_);
    const int inner_size = inputs[c]->size(this->axis_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        kernel_concatenate_forward, this->outer_size_ * inner_size,
        this->inner_total_size_, inner_size, inner_offset, x, y);
    inner_offset += inner_size;
  }
}

template <class T>
void ConcatenateCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  int inner_offset = 0;
  for (int c = 0; c < inputs.size(); ++c) {
    const int inner_size = inputs[c]->size(this->axis_);
    if (propagate_down[c]) {
      Tc *dx = inputs[c]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[c]);
      if (accum[c]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_concatenate_backward<Tc, true>),
                                       this->outer_size_ * inner_size,
                                       this->inner_total_size_, inner_size,
                                       inner_offset, dy, dx);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_concatenate_backward<Tc, false>),
                                       this->outer_size_ * inner_size,
                                       this->inner_total_size_, inner_size,
                                       inner_offset, dy, dx);
      }
    }
    inner_offset += inner_size;
  }
}
}
