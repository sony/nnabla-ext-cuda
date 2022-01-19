// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/tile.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_tile_forward(const int size, const int *idx,
                                    const T *x_data, T *y_data) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { y_data[i] = x_data[idx[i]]; }
}

template <typename T>
__global__ void kernel_tile_backward(const int size, const int *idx,
                                     const T *y_grad, T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { atomic_add(x_grad + idx[i], y_grad[i]); }
}

template <typename T>
void TileCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Tile<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  // Implicitly copy index map to device memory.
  (&this->idxmap_)->get(get_dtype<int>(), this->ctx_);
}

template <typename T>
void TileCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);
  auto src = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto arr = (&this->idxmap_)->get(get_dtype<int>(), this->ctx_);
  auto idx = arr->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_tile_forward, this->idxmap_.size(), idx,
                                 src, dst);
}

template <typename T>
void TileCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(this->device_);

  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }

  auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
  auto arr = (&this->idxmap_)->get(get_dtype<int>(), this->ctx_);
  auto idx = arr->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_tile_backward, this->idxmap_.size(),
                                 idx, g_y, g_x);
}
}
