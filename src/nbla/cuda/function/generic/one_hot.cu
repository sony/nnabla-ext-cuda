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

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/one_hot.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename TI, typename T>
__global__ void kernel_one_hot_forward(const int num, const int dim,
                                       const int size, const int *strides,
                                       const TI *x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    int addr = 0;
    for (int i2 = dim - 1; i2 >= 0; --i2) {
      addr += x[idx * dim + i2] * strides[i2];
    }
    y[idx * size + addr] = 1;
  }
}

template <typename TI, typename T>
void OneHotCuda<TI, T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  OneHot<TI, T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  const Shape_t stride = outputs[0]->strides();
  const int stride_info_size = stride.size() - (inputs[0]->ndim() - 1);
  Context cpu_ctx{{}, "CpuCachedArray", "0"};
  this->stride_info_buf_.reshape(Shape_t{stride_info_size}, true);
  int *stride_info_cpu =
      this->stride_info_buf_.cast(dtypes::INT, cpu_ctx, true)
          ->template pointer<int>();
  std::copy(stride.cbegin() + inputs[0]->ndim() - 1, stride.cend(),
            stride_info_cpu);
}

template <typename TI, typename T>
void OneHotCuda<TI, T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  const TIcu *x = inputs[0]->get_data_pointer<TIcu>(this->ctx_);
  outputs[0]->data()->zero();
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, false);
  const int *stride_info_gpu =
      this->stride_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_one_hot_forward, this->num_, this->dim_,
                                 this->size_, stride_info_gpu, x, y);
}

template <typename TI, typename T>
void OneHotCuda<TI, T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[0], error_code::value,
             "Index array can not be propagated down.");
}
}
