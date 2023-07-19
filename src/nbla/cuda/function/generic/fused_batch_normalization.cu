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
#include <nbla/cuda/function/fused_batch_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/add2.cuh>
#include <nbla/cuda/utils/relu.cuh>

namespace nbla {

namespace fused_batch_normalization_cuda {
template <typename T>
void relu_backward(const Size_t size, T *dx, const T *dy, const T *y) {
  const Size_t size2 = interpret_size<T>(size);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T((kernel_relu_backward<false>), size2,
                                        size, dx, y, dy);
}

template <typename T>
void add2_backward(const Size_t size, T *dx1, const T *dx, bool accum) {
  if (accum) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<T, true>), size, dx1,
                                   dx);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_add2_backward<T, false>), size, dx1,
                                   dx);
  }
}
} // namespace fused_batch_normalization_cuda

template <class T>
void FusedBatchNormalizationCuda<T>::relu_add2_backward(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum,
    Variable &relu_buf) {
  cuda_set_device(this->device_);

  // 1. Perform ReLU backward
  bool prop_down_add2 = (inputs.size() == 6 && propagate_down[5]);
  bool prop_down_bn =
      std::accumulate(propagate_down.begin(), propagate_down.begin() + 3, false,
                      std::logical_or<bool>());
  auto y = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto dx = relu_buf.cast_grad_and_get_pointer<Tcu>(this->ctx_);
  auto dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto size = outputs[0]->size();
  if (prop_down_add2 || prop_down_bn) {
    fused_batch_normalization_cuda::relu_backward<Tcu>(size, dx, dy, y);
  }

  // 2. Perform Add2 backward
  // NOTE: Output buffer for the first operand of the addition are re-used by
  // inplacing,
  // nothing done for it.
  if (prop_down_add2) {
    auto dx1 = inputs[5]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
    fused_batch_normalization_cuda::add2_backward<Tcu>(size, dx1, dx, accum[5]);
  }
}
} // namespace nbla
