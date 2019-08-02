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

// softmax_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/softmax.hpp>
#include <nbla/cuda/function/softmax_cross_entropy.hpp>
#include <nbla/cuda/limits.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, typename Tl>
__global__ void
kernel_softmax_cross_entropy_forward(const int size0x2_, const int size1_,
                                     const int size2_, const T *log_p,
                                     const Tl *l, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size2_ + i2;
    Tl label = l[j];
    const int k = i0 * size1_ * size2_ + label * size2_ + i2;
    y[j] = -log_p[k];
  }
}

template <typename T, typename Tl, bool accum>
__global__ void
kernel_softmax_cross_entropy_backward(const int size0x2_, const int size1_,
                                      const int size2_, const T *log_p,
                                      const T *dy, const Tl *l, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size2_ + i2;
    Tl label = l[j];
    T grad = dy[j];
    for (int i1 = 0; i1 < size1_; ++i1) {
      const int k = i0 * size1_ * size2_ + i1 * size2_ + i2;
      dx[k] = (accum ? dx[k] : (T)0) +
              grad * (std::exp(log_p[k]) - static_cast<int>(label == i1));
    }
  }
}

template <typename T, typename Tl>
void SoftmaxCrossEntropyCuda<T, Tl>::setup_impl(const Variables &inputs,
                                                const Variables &outputs) {
  SoftmaxCrossEntropy<T>::setup_impl(inputs, outputs);
}

template <typename T, typename Tl>
void SoftmaxCrossEntropyCuda<T, Tl>::forward_impl(const Variables &inputs,
                                                  const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Variable &tso = this->log_softmax_output_;
  this->log_softmax_->forward(Variables{inputs[0]}, Variables{&tso});
  // Setting up variables
  const Tc *log_p = tso.get_data_pointer<Tc>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_softmax_cross_entropy_forward,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, log_p, l, y);
}

template <typename T, typename Tl>
void SoftmaxCrossEntropyCuda<T, Tl>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[1], error_code::value,
             "Label can not be propagated down.");
  if (!propagate_down[0])
    return;

  cuda_set_device(std::stoi(this->ctx_.device_id));
  Variable &tso = this->log_softmax_output_;
  const Tc *log_p = tso.get_data_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_softmax_cross_entropy_backward<Tc, Tl, true>),
        this->size0_ * this->size2_, this->size1_, this->size2_, log_p, dy, l,
        dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_softmax_cross_entropy_backward<Tc, Tl, false>),
        this->size0_ * this->size2_, this->size1_, this->size2_, log_p, dy, l,
        dx);
  }
}
}
