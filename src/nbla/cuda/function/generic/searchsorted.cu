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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/searchsorted.hpp>
#include <nbla/variable.hpp>

// TODO: Remove these #includes. Only for debug.
#include <iostream>
#include <typeinfo>

namespace nbla {

template <typename T>
void SearchSortedCuda<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {

  SearchSorted<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
__global__ void
kernel_searchsorted_forward(const int v_size, const int ss_last_dim_,
                            const int v_last_dim_, const int inner_size_,
                            const T *sorted_sequence, const T *values, T *y,
                            bool right_) {
  NBLA_CUDA_KERNEL_LOOP(idx, v_size) {
    const int i = idx / v_last_dim_;
    const int j = idx % v_last_dim_;

    size_t v_idx = i * v_last_dim_ + j;
    size_t start = i * ss_last_dim_, end = (i + 1) * ss_last_dim_ - 1;
    size_t i_idx;

    while (1) {

      if (values[v_idx] > sorted_sequence[end]) {
        i_idx = end + 1;
        break;
      }

      if (right_ && values[v_idx] == sorted_sequence[end]) {
        i_idx = end + 1;
        break;
      }

      if (values[v_idx] < sorted_sequence[start]) {
        i_idx = start;
        break;
      }

      if (!right_ and values[v_idx] == sorted_sequence[start]) {
        i_idx = start;
        break;
      }

      if (end - start <= 1) {
        i_idx = end;
        break;
      }
      size_t mid =
          (start + end) % 2 == 0 ? (start + end) / 2 : (start + end + 1) / 2;

      bool check_condition = right_ ? (values[v_idx] < sorted_sequence[mid])
                                    : (values[v_idx] <= sorted_sequence[mid]);
      if (check_condition)
        end = mid;
      else
        start = mid;
    }

    y[v_idx] = i_idx - i * ss_last_dim_;
  }
}

template <typename T>
void SearchSortedCuda<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu *sorted_sequence = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *values = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_searchsorted_forward, inputs[1]->size(),
                                 this->ss_last_dim_, this->v_last_dim_,
                                 this->inner_size_, sorted_sequence, values, y,
                                 this->right_);
}

template <typename T>
void SearchSortedCuda<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {}
}
