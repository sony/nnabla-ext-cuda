// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/function/embed.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, typename T1>
__global__ void kernel_embed_forward(const int num, T1 *y, const T *x,
                                     const T1 *w, int stride0) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    const int i = idx / stride0;
    const int j = idx % stride0;
    y[idx] = w[x[i] * stride0 + j];
  }
}

template <typename T, typename T1, typename Tw>
__global__ void kernel_embed_backward_weight(const int num, Tw *dw, const T *x,
                                             const T1 *dy, int stride0) {
  // TODO: optimize
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    const int i = idx / stride0;
    const int j = idx % stride0;
    atomicAdd(dw + x[i] * stride0 + j,
              (typename CudaTypeForceFloat<T1>::type)dy[i * stride0 + j]);
  }
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  Embed<T, T1>::setup_impl(inputs, outputs);
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  typedef typename CudaType<T1>::type Tc;
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const Tc *w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  Size_t stride0 = inputs[1]->size(1);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_embed_forward,
                                 inputs[0]->size() * stride0, y, x, w, stride0);
}

template <typename T, typename T1>
void EmbedCuda<T, T1>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  typedef typename CudaType<T1>::type Tc;
  // atomicAdd doesn't support half precision. Force to use float32 instead.
  typedef typename CudaTypeForceFloat<T1>::type Tw;

  NBLA_CHECK(!propagate_down[0], error_code::value,
             "Index array can not be propagated down.");
  if (!propagate_down[1]) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (!accum[1])
    inputs[1]->grad()->zero();
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  Tw *dw = inputs[1]->cast_grad_and_get_pointer<Tw>(this->ctx_, false);
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);

  Size_t stride0 = inputs[1]->size(1);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_embed_backward_weight,
                                 inputs[0]->size() * stride0, dw, x, dy,
                                 stride0);
}
}
