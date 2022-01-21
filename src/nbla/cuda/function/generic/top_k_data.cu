// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/top_k_data.hpp>
#include <nbla/cuda/utils/top_k.cuh>
#include <nbla/variable.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace nbla {

namespace top_k_data {

template <bool REDUCE, typename T>
__global__ void copy_index_and_value(const int k,
                                     const unsigned int *sorted_idx, const T *x,
                                     T *y, unsigned int *top_k_idx) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted_idx[i];
    y[REDUCE ? i : idx] = x[idx];
    top_k_idx[i] = idx;
  }
}

template <bool REDUCE, typename T>
__global__ void copy_index_and_value(const int k, const ValIdx<T> *sorted,
                                     const T *x, T *y,
                                     unsigned int *top_k_idx) {
  NBLA_CUDA_KERNEL_LOOP(i, k) {
    const auto idx = sorted[i].index();
    y[REDUCE ? i : idx] = x[idx];
    top_k_idx[i] = idx;
  }
}

template <typename T> __global__ void set_to_zero(const int size, T *data) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { data[i] = 0; }
}

template <typename T> __global__ void set_to_absolute(const int size, T *data) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { data[i] = abs(data[i]); }
}

template <typename T>
__global__ void add_gradient(const int size, const T *g_y, T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { g_x[i] += g_y[i]; }
}

template <typename T>
__global__ void add_gradient(const int size, const unsigned int *idx,
                             const T *g_y, T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { g_x[idx[i]] += g_y[i]; }
}

template <typename T>
__global__ void set_gradient(const int size, const T *g_y, T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { g_x[i] = g_y[i]; }
}

template <typename T>
__global__ void set_gradient(const int size, const unsigned int *idx,
                             const T *g_y, T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { g_x[idx[i]] = g_y[i]; }
}
} // namspace top_k_data

template <typename T>
void TopKDataCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  TopKData<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (this->k_ > 1024) {
    this->buffer_.reshape(Shape_t{this->ss_}, true);
  } else {
    this->buffer_.reshape(Shape_t{static_cast<Size_t>(sizeof(Buffer<Tcu>))},
                          true);
  }
}

template <typename T>
void TopKDataCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  using namespace top_k_data;
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];

  auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto tk_idx =
      (reinterpret_cast<Variable &>(this->top_k_idx_)
           .cast_data_and_get_pointer<unsigned int>(this->ctx_, true));

  if (!this->reduce_)
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_to_zero, y->size(), y_data);

  if (this->k_ > 1024) {
    // For large K we use thrust sort_by_key to do a radix sort of
    // data and index. This is not very efficient but large K is not
    // he expected use case. The code could be splitting the input
    // into a smaller partition of the k-th largest values before
    // sorting.
    auto buffer_raw =
        this->buffer_.cast(get_dtype<unsigned int>(), this->ctx_, true)
            ->template pointer<unsigned int>();
    auto buffer_ptr = thrust::device_pointer_cast(buffer_raw);

    for (int s = 0; s < this->ns_; s++) {
      auto x_data_vec = thrust::device_vector<Tcu>(x_data, x_data + this->ss_);
      auto sorted_val = thrust::raw_pointer_cast(x_data_vec.data());
      auto sorted_idx = thrust::raw_pointer_cast(buffer_ptr);

      if (this->abs_) {
        auto raw_ptr = thrust::raw_pointer_cast(x_data_vec.data());
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_to_absolute, this->ss_, raw_ptr);
      }

      thrust::sequence(buffer_ptr, buffer_ptr + this->ss_);
      thrust::sort_by_key(x_data_vec.begin(), x_data_vec.end(), buffer_ptr,
                          thrust::greater<Tcu>());

      if (this->reduce_) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_index_and_value<true>, this->k_,
                                       sorted_idx, x_data, y_data, tk_idx);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_index_and_value<false>, this->k_,
                                       sorted_idx, x_data, y_data, tk_idx);
      }
      x_data += this->ss_; // increase by input sample size
      y_data += this->fs_; // increase by output feature size
      tk_idx += this->k_;
    }
  } else {
    auto buffer = this->buffer_.cast(get_dtype<char>(), this->ctx_, true)
                      ->template pointer<Buffer<Tcu>>();

    for (int s = 0; s < this->ns_; s++) {
      if (this->abs_) {
        top_k<Tcu, true>(x_data, this->ss_, this->k_, buffer);
      } else {
        top_k<Tcu, false>(x_data, this->ss_, this->k_, buffer);
      }
      if (this->reduce_) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_index_and_value<true>, this->k_,
                                       &buffer->sorted[0], x_data, y_data,
                                       tk_idx);
      } else {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(copy_index_and_value<false>, this->k_,
                                       &buffer->sorted[0], x_data, y_data,
                                       tk_idx);
      }
      x_data += this->ss_; // increase by input sample size
      y_data += this->fs_; // increase by output feature size
      tk_idx += this->k_;
    }
  }
  this->forward_done_ = true;
}

template <typename T>
void TopKDataCuda<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum_gradient) {
  if (!(propagate_down[0]))
    return;

  NBLA_CHECK(this->forward_done_, error_code::value,
             "Forward must be called before calling backward.");

  using namespace top_k_data;
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto y = outputs[0];

  auto g_y = y->get_grad_pointer<Tcu>(this->ctx_);
  auto idx = (reinterpret_cast<Variable &>(this->top_k_idx_)
                  .get_data_pointer<unsigned int>(this->ctx_));

  if (this->reduce_) {
    if (accum_gradient[0]) {
      auto g_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      for (int s = 0; s < this->ns_; s++) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(add_gradient, this->k_, idx, g_y, g_x);
        g_x += this->ss_;
        g_y += this->fs_;
        idx += this->k_;
      }
    } else {
      auto g_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_to_zero, x->size(), g_x);
      for (int s = 0; s < this->ns_; s++) {
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_gradient, this->k_, idx, g_y, g_x);
        g_x += this->ss_;
        g_y += this->fs_;
        idx += this->k_;
      }
    }
  } else {
    if (accum_gradient[0]) {
      auto g_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(add_gradient, x->size(), g_y, g_x);
    } else {
      auto g_x = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(set_gradient, x->size(), g_y, g_x);
    }
  }
}

} // namespace nnabla
