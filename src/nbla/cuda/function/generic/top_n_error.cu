// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/top_n_error.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, typename T1>
__global__ void kernel_top_n_error_reduction(const int num, const int size1_,
                                             const int size2_, const int n_,
                                             const T *x, const T1 *l, T *y) {

  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    const int idx2 = idx % size2_;
    const int idx0 = idx / size2_;

    const int l_in = idx0 * size2_ + idx2;
    const T1 label = l[l_in];
    const int x_in = idx0 * size2_ * size1_ + idx2;
    const T threshold = x[x_in + label * size2_];

    T1 count = 0;
    for (int i = 0; i < size1_; i++) {
      count += x[x_in + i * size2_] >= threshold;
    }
    y[l_in] = count > n_;
  }
}

template <typename T, typename T1>
void TopNErrorCuda<T, T1>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *p = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const T1 *l = inputs[1]->get_data_pointer<T1>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_top_n_error_reduction,
                                 this->size0_ * this->size2_, this->size1_,
                                 this->size2_, this->n_, p, l, y);
}

template <typename T, typename T1>
void TopNErrorCuda<T, T1>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  // not supported
}
}
