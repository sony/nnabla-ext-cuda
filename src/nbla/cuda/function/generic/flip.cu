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

template <typename T>
void FlipCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Flip<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  const Shape_t shape = outputs[0]->shape();
  const Shape_t stride = outputs[0]->strides();
  size_t size = outputs[0]->size();
  const int shape_info_size = shape.size() * 3;

  Context cpu_ctx{{}, "CpuCachedArray", "0"};
  this->shape_info_buf_.reshape(Shape_t{shape_info_size}, true);
  int *shape_info_cpu = this->shape_info_buf_.cast(dtypes::INT, cpu_ctx, true)
                            ->template pointer<int>();
  for (int i = 0; i < shape.size(); i++) {
    shape_info_cpu[i * 3] = shape[i];      // shape
    shape_info_cpu[i * 3 + 1] = stride[i]; // stride
    auto itr = std::find(this->axes_.begin(), this->axes_.end(), i);
    shape_info_cpu[i * 3 + 2] = itr != this->axes_.end(); // flip
  }
}

template <typename T, bool accum>
__global__ void kernel_flip(const int num, const int dim, T *y, const T *x,
                            const int *shape_info) {

  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    // Determin input address
    int addr = 0;
    for (int id = 0; id < dim; id++) {
      const int shape_info_offset = id * 3;
      const int o = (idx / shape_info[shape_info_offset + 1]) // stride
                    % shape_info[shape_info_offset];          // shape
      const int i = shape_info[shape_info_offset + 2] ?       // flip
                        shape_info[shape_info_offset] - 1 - o
                                                      : o;
      addr += i * shape_info[shape_info_offset + 1]; // stride
    }
    // Copy input to output
    if (accum) {
      y[idx] += x[addr];
    } else {
      y[idx] = x[addr];
    }
  }
}

template <typename T>
void FlipCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  size_t size = outputs[0]->size();
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_flip<Tcu, false>), size,
                                 inputs[0]->ndim(), y, x, shape_info_gpu);
}

template <typename T>
void FlipCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  cuda_set_device(this->device_);
  if (!propagate_down[0]) {
    return;
  }

  Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  size_t size = outputs[0]->size();
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();

  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_flip<Tcu, true>), size,
                                   inputs[0]->ndim(), dx, dy, shape_info_gpu);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_flip<Tcu, false>), size,
                                   inputs[0]->ndim(), dx, dy, shape_info_gpu);
  }
}
}
