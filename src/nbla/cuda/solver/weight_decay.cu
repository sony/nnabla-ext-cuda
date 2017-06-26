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

#include <nbla/cuda/common.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_weight_decay(const int num, T *grad, const T *data,
                                    const float decay_rate) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] += decay_rate * data[idx]; }
}

template <typename T>
void weight_decay_cuda(const Context &ctx, const shared_ptr<Variable> param,
                       float decay_rate) {
  cuda_set_device(std::stoi(ctx.device_id));
  Size_t size = param->size();
  const T *data = param->get_data_pointer<T>(ctx);
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_weight_decay, size, grad, data,
                                 decay_rate);
}

// Template instantiation
template void weight_decay_cuda<float>(const Context &ctx,
                                       const shared_ptr<Variable> param,
                                       float decay_rate);
}
