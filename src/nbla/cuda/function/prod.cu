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
#include <nbla/cuda/function/prod.hpp>
#include <nbla/cuda/utils/device_reduce.cuh>
#include <nbla/cuda/utils/reduce_ops/prod.cuh>

namespace nbla {

template <typename T>
void ProdCuda<T>::forward_impl_reduce(const T *x, T *y, int outer_size,
                                      int reduction_size) {
  cuda_set_device(this->device_);
  // TODO: Auto tune.
  if (reduction_size / outer_size < 32) {
    reduce_2d_mixed_parallel(outer_size, reduction_size, ProdOp<T>(x, y));
    return;
  }

  // Get block reduce buffer
  auto fbuff = cuda_get_reduction_buffer<T>(reduction_size, this->ctx_);
  ProdOp<T> pre_op(x, fbuff.second);
  ProdOp<T> post_op(fbuff.second, y);
  reduce_2d_parallel_reduction(outer_size, reduction_size, pre_op, post_op);
}

template <typename T, bool accum>
__global__ void kernel_reduce_prod_backward(const int num, int reduction_size,
                                            const T *dy, const T *x, const T *y,
                                            T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    int o = idx / reduction_size;
    if (accum) {
      dx[idx] += x[idx] == 0 ? 0 : dy[o] * y[o] / x[idx];
    } else {
      dx[idx] = x[idx] == 0 ? 0 : dy[o] * y[o] / x[idx];
    }
  }
}

template <typename T>
void ProdCuda<T>::backward_impl_reduce_prod(const T *dy, const T *x, const T *y,
                                            T *dx, int outer_size,
                                            int reduction_size, bool accum) {
  cuda_set_device(this->device_);
  if (accum) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_prod_backward<T, true>),
                                   outer_size * reduction_size, reduction_size,
                                   dy, x, y, dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_reduce_prod_backward<T, false>),
                                   outer_size * reduction_size, reduction_size,
                                   dy, x, y, dx);
  }
}

// template instantiation
template class ProdCuda<float>;
}
