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
#include <nbla/cuda/function/max.hpp>
#include <nbla/cuda/utils/device_reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/max.cuh>

namespace nbla {

template <typename T>
void MaxCuda<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                     int reduction_size) {
  cuda_set_device(this->device_);
  VariablePtr vind = this->index_buff_;
  int *ind = vind->cast_data_and_get_pointer<int>(this->ctx_);

  // TODO: Auto tune.
  if (reduction_size / outer_size < 32) {
    reduce_2d_mixed_parallel(outer_size, reduction_size,
                             MaxPreOp<T>(x, y, ind));
    return;
  }

  // Get block reduce buffer
  auto fbuff = cuda_get_reduction_buffer<T>(reduction_size, this->ctx_);
  auto ibuff = cuda_get_reduction_buffer<int>(reduction_size, this->ctx_);
  MaxPreOp<T> pre_op(x, fbuff.second, ibuff.second);
  MaxPostOp<T> post_op(fbuff.second, ibuff.second, y, ind);
  reduce_2d_parallel_reduction(outer_size, reduction_size, pre_op, post_op);
}

template <typename T>
__global__ void kernel_reduce_index_backward(const int num, T *dx,
                                             const int *ind, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { dx[ind[idx]] += dy[idx]; }
}

template <typename T>
void MaxCuda<T>::backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                      int reduction_size, bool accum) {
  cuda_set_device(this->device_);
  if (!accum) {
    cudaMemsetAsync(dx, 0, sizeof(*dx) * outer_size * reduction_size);
  }
  VariablePtr vind = this->index_buff_;
  const int *ind = vind->get_data_pointer<int>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_index_backward, outer_size, dx,
                                 ind, dy);
}

// template instantiation
template class MaxCuda<float>;
}
