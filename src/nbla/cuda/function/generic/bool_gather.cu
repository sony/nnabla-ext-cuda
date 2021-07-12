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
#include <nbla/cuda/function/bool_gather.hpp>
#include <nbla/cuda/function/utils/bool_indexing.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void BoolGatherCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  BoolGather<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void BoolGatherCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);

  // Outputs
  auto mshape = inputs[1]->shape();
  auto B = inputs[1]->size();
  auto nnz = outputs[0]->shape()[0];
  auto D = outputs[0]->size() / nnz;
  Tcu *sdata = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  const Tcu *gdata = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  auto kernel = bool_indexing_cuda::kernel_bool_gather<Tcu>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, D, B, nnz, sdata, gdata, mask);
}

template <typename T>
void BoolGatherCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  auto mshape = inputs[1]->shape();
  auto B = inputs[1]->size();
  auto nnz = outputs[0]->shape()[0];
  auto D = outputs[0]->size() / nnz;
  const Tcu *g_sdata = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  const Tcu *mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  if (propagate_down[0]) {
    auto g_gdata =
        inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    auto kernel = accum[0]
                      ? bool_indexing_cuda::kernel_bool_scatter<Tcu, true>
                      : bool_indexing_cuda::kernel_bool_scatter<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, D, B, nnz, g_gdata, g_sdata, mask);
  }
}
}
