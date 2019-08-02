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
#include <nbla/cuda/function/min.hpp>
#include <nbla/cuda/utils/device_reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/min.cuh>

namespace nbla {

namespace {

template <typename T>
__global__ void adjust_index(const int size, T *data,
                             const int reduction_size) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { data[i] -= i * reduction_size; }
}
}

template <typename T>
void MinCuda<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  Min<T>::forward_impl(inputs, outputs);
  if (this->with_index_ || this->only_index_) {
    Variable *idx_var = this->only_index_ ? outputs[0] : outputs[1];
    auto idx_ptr = idx_var->cast_data_and_get_pointer<size_t>(this->ctx_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(adjust_index, idx_var->size(), idx_ptr,
                                   this->reduction_size_);
  }
}

template <typename T>
void MinCuda<T>::forward_impl_reduce(const T *x_, T *y_, int outer_size,
                                     int reduction_size) {
  const Tc *x = reinterpret_cast<const Tc *>(x_);
  Tc *y = reinterpret_cast<Tc *>(y_);
  cuda_set_device(this->device_);
  VariablePtr vind = this->index_buff_;
  int *ind = vind->cast_data_and_get_pointer<int>(this->ctx_, true);

  // TODO: Auto tune.
  if (reduction_size / outer_size < 32) {
    reduce_2d_mixed_parallel(outer_size, reduction_size,
                             MinPreOp<Tc>(x, y, ind));
    return;
  }

  // Get block reduce buffer
  auto fbuff = cuda_get_reduction_buffer<Tc>(reduction_size, this->ctx_);
  auto ibuff = cuda_get_reduction_buffer<int>(reduction_size, this->ctx_);
  MinPreOp<Tc> pre_op(x, fbuff.second, ibuff.second);
  MinPostOp<Tc> post_op(fbuff.second, ibuff.second, y, ind);
  reduce_2d_parallel_reduction(outer_size, reduction_size, pre_op, post_op);
}

template <typename T>
__global__ void kernel_reduce_index_backward(const int num, T *dx,
                                             const int *ind, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[ind[idx]] += dy[idx]; }
}

template <typename T>
void MinCuda<T>::backward_impl_reduce(const T *dy_, T *dx_, int outer_size,
                                      int reduction_size, bool accum) {
  const Tc *dy = reinterpret_cast<const Tc *>(dy_);
  Tc *dx = reinterpret_cast<Tc *>(dx_);
  cuda_set_device(this->device_);
  if (!accum) {
    cudaMemsetAsync(dx, 0, sizeof(*dx) * outer_size * reduction_size);
  }
  VariablePtr vind = this->index_buff_;
  const int *ind = vind->get_data_pointer<int>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_index_backward, outer_size, dx,
                                 ind, dy);
}
}
