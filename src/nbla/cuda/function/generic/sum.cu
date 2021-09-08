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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/function/sum.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/sum.cuh>

namespace nbla {

template <typename T>
void SumCuda<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  cuda_set_device(this->device_);
  Sum<T>::setup_impl(inputs, outputs);

  const Shape_t axes(this->axes_.cbegin(), this->axes_.cend());
  reduce_setup_(inputs[0]->shape(), axes);
}

template <typename T>
void SumCuda<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tc *const x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  device_sum(this->ctx_, x, y, reduce_setup_);
}

template <typename T, bool accum>
__global__ void kernel_reduce_sum_backward(const int num, T *dx, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[idx] = (accum ? dx[idx] : (T)0) + *dy; }
}

template <typename T>
void SumCuda<T>::backward_impl_reduce(const T *dy_, T *dx_, int outer_size,
                                      int reduction_size, bool accum) {
  const Tc *dy = reinterpret_cast<const Tc *>(dy_);
  Tc *dx = reinterpret_cast<Tc *>(dx_);
  cuda_set_device(this->device_);
  if (outer_size == 1) {
    if (accum) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_sum_backward<Tc, true>),
                                     reduction_size, dx, dy);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_sum_backward<Tc, false>),
                                     reduction_size, dx, dy);
    }
    return;
  }
  const Tc *ones =
      static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
          reduction_size, get_dtype<Tc>(), this->ctx_));
  cuda_gemm<Tc>(this->device_, dx, true, dy, outer_size, 1, false, ones, 1,
                reduction_size, false, 1, accum ? 1 : 0);
}
}
