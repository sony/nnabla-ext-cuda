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
#include <nbla/cuda/function/batch_logdet.hpp>
#include <nbla/variable.hpp>

#include "kernel/batch_det.cu"

namespace nbla {

template <typename T>
void BatchLogdetCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  BatchLogdet<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  batch_size_ = inputs[0]->shape()[0];
  dim_ = inputs[0]->shape()[1];
}

template <typename T>
void BatchLogdetCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  batch_det_forward<T, Tcu, true /* with_abs_log */>(
      this->ctx_, this->device_, inputs, outputs, this->dim_,
      this->batch_size_);
}

template <typename T>
void BatchLogdetCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  cuda_set_device(this->device_);
  BatchLogdet<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
