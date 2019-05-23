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

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/* UNDER REVIEW.

   NOTE: cudaMemcpy and kernel execution bat setup_impl.
*/
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/flip.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

__global__ void kernel_flip_create_table(const int num, const int dim,
                                         int *addr_table_buf,
                                         const int *shape_info_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    int addr = 0;
    for (int id = 0; id < dim; id++) {
      const int shape_info_offset = id * 3;
      const int o = (idx / shape_info_buf[shape_info_offset + 1]) // stride
                    % shape_info_buf[shape_info_offset];          // shape
      const int i = shape_info_buf[shape_info_offset + 2] ?       // flip
                        shape_info_buf[shape_info_offset] - 1 - o
                                                          : o;
      addr += i * shape_info_buf[shape_info_offset + 1]; // stride
    }
    addr_table_buf[idx] = addr;
  }
}

template <typename T>
void FlipCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Flip<T>::setup_impl(inputs, outputs);

  // Prepare address table
  const Shape_t shape = outputs[0]->shape();
  const Shape_t stride = outputs[0]->strides();
  size_t size = outputs[0]->size();
  this->addr_table_.reshape(shape, true);
  const int shape_info_size = shape.size() * 3;
  // shape, stride, flip
  int *shape_info = new int[shape_info_size];
  for (int i = 0; i < shape.size(); i++) {
    shape_info[i * 3] = shape[i];
    shape_info[i * 3 + 1] = stride[i];
    auto itr = std::find(this->axes_.begin(), this->axes_.end(), i);
    shape_info[i * 3 + 2] = itr != this->axes_.end();
  }
  Shape_t shape_info_shape;
  shape_info_shape.push_back(shape_info_size);
  Variable shape_info_variable;
  shape_info_variable.reshape(shape_info_shape, true);
  int *shape_info_buf =
      shape_info_variable.cast_data_and_get_pointer<int>(this->ctx_, true);
  cudaMemcpy(shape_info_buf, shape_info, sizeof(int) * shape_info_size,
             cudaMemcpyHostToDevice);
  delete[] shape_info;
  Variable *addr_table_ = &this->addr_table_;
  int *addr_table_buf =
      addr_table_->cast_data_and_get_pointer<int>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_flip_create_table, size, shape.size(),
                                 addr_table_buf, shape_info_buf);
}

template <typename T>
__global__ void kernel_flip_forward(const int num, T *y, const T *x,
                                    const int *addr_table_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = x[addr_table_buf[idx]]; }
}

template <typename T>
void FlipCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const int *addr_table_buf =
      this->addr_table_.template get_data_pointer<int>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  size_t size = outputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_flip_forward, size, y, x,
                                 addr_table_buf);
}

template <typename T, bool accum>
__global__ void kernel_flip_backward(const int num, T *dx, const T *dy,
                                     const int *addr_table_buf) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    T &ref = dx[addr_table_buf[idx]];
    ref = (accum ? ref : (T)0) + dy[idx];
  }
}

template <typename T>
void FlipCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
  const int *addr_table_buf =
      this->addr_table_.template get_data_pointer<int>(this->ctx_);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  size_t size = outputs[0]->size();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_flip_backward<Tc, true>), size, dx,
                                   dy, addr_table_buf);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_flip_backward<Tc, false>), size, dx,
                                   dy, addr_table_buf);
  }
}
}
