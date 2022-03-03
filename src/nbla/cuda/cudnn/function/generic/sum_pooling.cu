// Copyright 2020,2021 Sony Corporation.
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
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/sum_pooling.hpp>
#include <nbla/variable.hpp>

namespace nbla {

namespace sum_pooling {

template <typename T>
__global__ void pool_multiply(const int size, T *y, const float pool_size) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] *= pool_size; }
}

template <typename T, bool accum = true>
__global__ void kernel_accum(const int size, T *dx, const T *dx_tmp) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    dx[idx] = accum ? dx[idx] + dx_tmp[idx] : dx_tmp[idx];
  }
}
}

template <typename T>
void SumPoolingCudaCudnn<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  // TODO: implement
  NBLA_CHECK(
      this->ignore_border_, error_code::not_implemented,
      "CudnnSumPoolingCudaCudnn with (ignore_border=False) is not supported.");
  this->average_pooling_.setup(inputs, outputs);
  pool_size_ = std::accumulate(this->kernel_.begin(), this->kernel_.end(), 1,
                               std::multiplies<int>());
}

template <typename T>
void SumPoolingCudaCudnn<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  // Forward of AveragePool
  this->average_pooling_.forward(inputs, outputs);
  // Multiply results by pool_size
  auto size = outputs[0]->size();
  auto y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((sum_pooling::pool_multiply<Tcu>), size, y,
                                 pool_size_);
}

template <typename T>
void SumPoolingCudaCudnn<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  // Seem to have to call the cudnn average pooling with the same pointer
  // addresses,
  // so the redundant logics are in here
  auto size = inputs[0]->size();
  if (accum[0]) {
    // Save gradients once for accumulation case
    auto iv_sptr = make_shared<Variable>(inputs[0]->shape());
    auto iv = iv_sptr.get();
    auto ig0 = iv->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto dx0 = inputs[0]->get_grad_pointer<Tcu>(this->ctx_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((sum_pooling::kernel_accum<Tcu, false>),
                                   size, ig0, dx0);
    // Backward of AveragePool
    this->average_pooling_.backward(inputs, outputs, propagate_down, {false});
    // Multiply results by pool_size
    auto dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((sum_pooling::pool_multiply<Tcu>), size, dx,
                                   pool_size_);
    // Accumulate
    auto ig = iv->get_data_pointer<Tcu>(this->ctx_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((sum_pooling::kernel_accum<Tcu, true>), size,
                                   dx, ig);
  } else {
    // Backward of AveragePool
    this->average_pooling_.backward(inputs, outputs, propagate_down, {false});
    // Multiply results by pool_size
    auto dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((sum_pooling::pool_multiply<Tcu>), size, dx,
                                   pool_size_);
  }
}
}
