// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
#include <nbla/cuda/function/top_k_grad.hpp>
#include <nbla/cuda/utils/top_k.cuh>
#include <nbla/variable.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace nbla {

namespace top_k_grad {

template <typename T> __global__ void set_to_zero(const int size, T *data) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { data[i] = 0; }
}

template <typename T> __global__ void set_to_absolute(const int size, T *data) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { data[i] = abs(data[i]); }
}

template <typename T>
__global__ void add_gradient(const int k, const ValIdx<T> *sorted,
                             const T *y_grad, T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted[i].index();
    x_grad[idx] += y_grad[idx];
  }
}

template <typename T>
__global__ void set_gradient(const int k, const ValIdx<T> *sorted,
                             const T *y_grad, T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted[i].index();
    x_grad[idx] = y_grad[idx];
  }
}

template <typename T>
__global__ void add_gradient(const int k, const unsigned int *sorted_idx,
                             const T *y_grad, T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted_idx[i];
    x_grad[idx] += y_grad[idx];
  }
}

template <typename T>
__global__ void set_gradient(const int k, const unsigned int *sorted_idx,
                             const T *y_grad, T *x_grad) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted_idx[i];
    x_grad[idx] = y_grad[idx];
  }
}

} // namspace top_k_grad

template <typename T>
void TopKGradCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  TopKGrad<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (this->k_ > 1024) {
    this->buffer_ =
        make_shared<CudaCachedArray>(outputs[0]->size(this->base_axis_),
                                     get_dtype<unsigned int>(), this->ctx_);
  } else {
    this->buffer_ = make_shared<CudaCachedArray>(sizeof(Buffer<Tcu>),
                                                 get_dtype<char>(), this->ctx_);
  }
}

template <typename T>
void TopKGradCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];

  auto x_data = x->data()->get(get_dtype<Tcu>(), this->ctx_);
  auto y_data = y->data()->cast(get_dtype<Tcu>(), this->ctx_, true);

  y_data->copy_from(x_data);
}

template <typename T>
void TopKGradCuda<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!(propagate_down[0]))
    return;

  using namespace top_k_grad;
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];

  auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);
  auto x_grad = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  auto idx = (reinterpret_cast<Variable &>(this->top_k_idx_)
                  .get_data_pointer<unsigned int>(this->ctx_));

  if (!accum[0])
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_to_zero, x->size(), x_grad);

  auto inner_size = y->size(this->base_axis_);
  auto outer_size = y->size() / inner_size;

  if (this->k_ > 1024) {
    // For large K we use thrust sort_by_key to do a radix sort of
    // data and index. This is not very efficient but large K is not
    // he expected use case. The code could be splitting the input
    // into a smaller partition of the k-th largest values before
    // sorting.
    auto buffer_raw = this->buffer_->pointer<unsigned int>();
    auto buffer_ptr = thrust::device_pointer_cast(buffer_raw);

    for (int s = 0; s < outer_size; s++) {
      auto y_grad_vec = thrust::device_vector<Tcu>(y_grad, y_grad + inner_size);
      auto sorted_val = thrust::raw_pointer_cast(y_grad_vec.data());
      auto sorted_idx = thrust::raw_pointer_cast(buffer_ptr);

      if (this->abs_) {
        auto raw_ptr = thrust::raw_pointer_cast(y_grad_vec.data());
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_to_absolute, inner_size, raw_ptr);
      }

      thrust::sequence(buffer_ptr, buffer_ptr + inner_size);
      thrust::sort_by_key(y_grad_vec.begin(), y_grad_vec.end(), buffer_ptr,
                          thrust::greater<Tcu>());

      if (accum[0]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(add_gradient, this->k_, sorted_idx,
                                       y_grad, x_grad);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_gradient, this->k_, sorted_idx,
                                       y_grad, x_grad);
      }
      y_grad += inner_size;
      x_grad += inner_size;
    }
  } else {
    auto buffer = this->buffer_->pointer<Buffer<Tcu>>();

    for (int s = 0; s < outer_size; s++) {
      if (this->abs_) {
        top_k<Tcu, true>(y_grad, inner_size, this->k_, buffer);
      } else {
        top_k<Tcu, false>(y_grad, inner_size, this->k_, buffer);
      }
      if (accum[0]) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(add_gradient, this->k_,
                                       &buffer->sorted[0], y_grad, x_grad);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_gradient, this->k_,
                                       &buffer->sorted[0], y_grad, x_grad);
      }
      y_grad += inner_size;
      x_grad += inner_size;
    }
  }
}
}
