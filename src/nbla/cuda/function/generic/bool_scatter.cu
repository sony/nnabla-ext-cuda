// Copyright 2021,2022 Sony Group Corporation.
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
#include <nbla/cuda/function/bool_scatter.hpp>
#include <nbla/cuda/function/utils/bool_indexing.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void BoolScatterCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  BoolScatter<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void BoolScatterCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  auto mshape = inputs[1]->shape();
  auto gshape = outputs[0]->shape();
  auto B = inputs[1]->size();
  auto nnz = inputs[0]->shape()[0];
  auto D = inputs[0]->size() / nnz;

  auto inplace = (inputs.size() > 2);

  auto sdata = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto gdata = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, !inplace);

  auto kernel =
      inplace ? bool_indexing_cuda::kernel_bool_scatter<Tcu, false, true>
              : bool_indexing_cuda::kernel_bool_scatter<Tcu, false, false>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, D, B, nnz, gdata, sdata, mask);
}

namespace bool_scatter_cuda {

template <typename T, bool accum = false>
__global__ void kernel_masked_identity(int B, int D, T *g_gdata_inp,
                                       const T *g_gdata_out, const T *mask) {
  NBLA_CUDA_KERNEL_LOOP(b, B) {
    auto umask_b = T(mask[b] == T(0));
    for (int d = 0; d < D; ++d) {
      if (accum)
        g_gdata_inp[b * D + d] += umask_b * g_gdata_out[b * D + d];
      else
        g_gdata_inp[b * D + d] = umask_b * g_gdata_out[b * D + d];
    }
  }
}
}

template <typename T>
void BoolScatterCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);

  auto mshape = inputs[1]->shape();
  auto gshape = outputs[0]->shape();
  auto B = inputs[1]->size();
  auto nnz = inputs[0]->shape()[0];
  auto D = inputs[0]->size() / nnz;

  auto g_gdata = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  if (propagate_down[0]) {
    auto g_sdata =
        inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    auto kernel = accum[0] ? bool_indexing_cuda::kernel_bool_gather<Tcu, true>
                           : bool_indexing_cuda::kernel_bool_gather<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, D, B, nnz, g_sdata, g_gdata, mask);
  }

  // inplace
  if (inputs.size() > 2 && propagate_down[2]) {
    auto g_gdata_inp =
        inputs[2]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[2]);
    auto kernel = accum[2]
                      ? bool_scatter_cuda::kernel_masked_identity<Tcu, true>
                      : bool_scatter_cuda::kernel_masked_identity<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, B, D, g_gdata_inp, g_gdata, mask);
  }
}
}
