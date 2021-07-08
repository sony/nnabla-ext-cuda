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

// categorical_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/categorical_cross_entropy.hpp>
#include <nbla/cuda/limits.hpp>
#include <nbla/function/categorical_cross_entropy.hpp>
#include <nbla/variable.hpp>

//#define CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0

namespace nbla {

#if CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0
// CategoricalCrossEntropyCuda forward kernel of parallelizing over size0_
template <typename T, typename Tl>
__global__ void kernel_categorical_cross_entropy_forward_naive(
    const int size0_, const int size1_, const int size2_, const T *p,
    const Tl *l, T *y) {
  NBLA_CUDA_KERNEL_LOOP(i0, size0_) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      Tl label = l[j];
      if (label < 0) {
        y[j] = 0;
        continue;
      }
      const int k = i0 * size1_ * size2_ + label * size2_ + i2;
      y[j] = -log(max(p[k], numeric_limits_cuda<T>::min()));
    }
  }
}

// CategoricalCrossEntropyCuda backward kernel of parallelizing over size0_
template <typename T, typename Tl>
__global__ void kernel_categorical_cross_entropy_backward_naive(
    const int size0_, const int size1_, const int size2_, const T *p,
    const T *dy, const Tl *l, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(i0, size0_) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      Tl label = l[j];
      if (label < 0) {
        continue;
      }
      const int k = i0 * size1_ * size2_ + label * size2_ + i2;
      dx[k] += -dy[j] / max(p[k], numeric_limits_cuda<T>::min());
    }
  }
}
#else // CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0_AND_SIZE2
// CategoricalCrossEntropyCuda forward kernel of parallelizing over size0_ *
// size2_
template <typename T, typename Tl>
__global__ void
kernel_categorical_cross_entropy_forward(const int size0x2_, const int size1_,
                                         const int size2_, const T *p,
                                         const Tl *l, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size2_ + i2;
    Tl label = l[j];
    if (label < 0) {
      y[j] = 0;
      continue;
    }
    const int k = i0 * size1_ * size2_ + label * size2_ + i2;
    y[j] = -log(max(p[k], numeric_limits_cuda<T>::min()));
  }
}

// CategoricalCrossEntropyCuda backward kernel of parallelizing over size0_ *
// size2_
template <typename T, typename Tl>
__global__ void
kernel_categorical_cross_entropy_backward(const int size0x2_, const int size1_,
                                          const int size2_, const T *p,
                                          const T *dy, const Tl *l, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;
    const int j = i0 * size2_ + i2;
    Tl label = l[j];
    if (label < 0) {
      continue;
    }
    const int k = i0 * size1_ * size2_ + label * size2_ + i2;
    dx[k] += -dy[j] / max(p[k], numeric_limits_cuda<T>::min());
  }
}
#endif

template <typename T, typename Tl>
void CategoricalCrossEntropyCuda<T, Tl>::setup_impl(const Variables &inputs,
                                                    const Variables &outputs) {
  CategoricalCrossEntropy<T>::setup_impl(inputs, outputs);
}

template <typename T, typename Tl>
void CategoricalCrossEntropyCuda<T, Tl>::forward_impl(
    const Variables &inputs, const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Setting up variables
  const Tc *p = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
#if CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_categorical_cross_entropy_forward_naive,
                                 this->size0_, this->size1_, this->size2_, p, l,
                                 y);
#else // CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0_AND_SIZE2
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_categorical_cross_entropy_forward,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, p, l, y);
#endif
}

template <typename T, typename Tl>
void CategoricalCrossEntropyCuda<T, Tl>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[1], error_code::value,
             "Label can not be propagated down.");
  if (!propagate_down[0])
    return;
  if (!accum[0])
    inputs[0]->grad()->zero();

  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *p = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
#if CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_categorical_cross_entropy_backward_naive, this->size0_,
      this->size1_, this->size2_, p, dy, l, dx);
#else // CATEGORICAL_CROSS_ENTROPY_CUDA_OF_PARALLEL_OVER_SIZE0_AND_SIZE2
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_categorical_cross_entropy_backward,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, p, dy, l, dx);
#endif
}
}
