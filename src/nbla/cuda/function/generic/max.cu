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
#include <nbla/cuda/utils/reduce_ops/max.cuh>

namespace nbla {

template <typename T>
void MaxCuda<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  cuda_set_device(this->device_);
  Max<T>::setup_impl(inputs, outputs);

  const Shape_t axes(this->axes_.cbegin(), this->axes_.cend());
  reduce_setup_(inputs[0]->shape(), axes);
}

template <typename T>
void MaxCuda<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  cuda_set_device(this->device_);

  // Inputs
  auto x = inputs[0]->get_data_pointer<Tc>(this->ctx_);

  // Outputs
  Variable out_buf(outputs[0]->shape()); // Intermediate buffer used if required
  Variable *idx_var;
  Variable *out_var;

  if (this->only_index_) {
    out_var = &out_buf;
    idx_var = outputs[0];
  } else if (this->with_index_) {
    out_var = outputs[0];
    idx_var = outputs[1];
  } else {
    out_var = outputs[0];
    idx_var = this->index_buff_.get();
  }

  auto y = out_var->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  auto idx = idx_var->cast_data_and_get_pointer<Size_t>(this->ctx_, true);

  // Max
  device_max(this->ctx_, x, y, idx, reduce_setup_);
}

template <typename T>
__global__ void kernel_reduce_index_backward(const int outer_size,
                                             const int reduction_size, T *dx,
                                             const Size_t *ind, const T *dy) {
  NBLA_CUDA_KERNEL_LOOP(o, outer_size) {
    dx[o * reduction_size + ind[o]] += dy[o];
  }
}

template <typename T>
void MaxCuda<T>::backward_impl_reduce(const T *dy_, T *dx_, int outer_size,
                                      int reduction_size, bool accum) {
  // ToDo: Use Size_t instead of int.
  const Tc *dy = reinterpret_cast<const Tc *>(dy_);
  Tc *dx = reinterpret_cast<Tc *>(dx_);
  cuda_set_device(this->device_);
  if (!accum) {
    cudaMemsetAsync(dx, 0, sizeof(*dx) * outer_size * reduction_size);
  }
  VariablePtr vind = this->index_buff_;
  // Use Size_t instead of int, matching the type with that used in forward_impl
  // to avoid the unnecessary data copy of type cast.
  const Size_t *ind = vind->get_data_pointer<Size_t>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_reduce_index_backward, outer_size,
                                 reduction_size, dx, ind, dy);
}
}
