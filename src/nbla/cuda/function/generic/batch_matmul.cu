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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/batch_matmul.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

namespace nbla {

/*
  NOTE: As a computational backend, cublas<t>gemmBatched is used if cublas<8,
  otherwise cublas<t>gemmStrdiedBatched
*/

// ----------------------------------------------------------------------
// With cublas<t>gemmStridedBatched
// ----------------------------------------------------------------------
#if CUDA_VERSION >= 8000
template <typename T>
void BatchMatmulCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<Tc>(this->ctx_);
  };
  // Broadcast
  Variable a_broadcast;
  Variable b_broadcast;
  if (this->f_broadcast_a_)
    execute(this->f_broadcast_a_, {inputs[0]}, {&a_broadcast});
  if (this->f_broadcast_b_)
    execute(this->f_broadcast_b_, {inputs[1]}, {&b_broadcast});
  // BMM
  const Tc *a =
      this->f_broadcast_a_ ? _get_data(&a_broadcast) : _get_data(inputs[0]);
  const Tc *b =
      this->f_broadcast_b_ ? _get_data(&b_broadcast) : _get_data(inputs[1]);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  cuda_gemm_strided_batched<Tc>(this->device_, y, true, a, this->col_a_,
                                this->row_a_, !this->transpose_a_, b,
                                this->col_b_, this->row_b_, !this->transpose_b_,
                                1, 0, this->samples_);
}

template <typename T>
void BatchMatmulCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<Tc>(this->ctx_);
  };
  auto _cast_grad = [this](Variable *v, bool wo) {
    return v->cast_grad_and_get_pointer<Tc>(this->ctx_, wo);
  };
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  if (propagate_down[0]) {
    // Broadcast
    Variable a_broadcast;
    Variable b_broadcast;
    if (this->f_broadcast_a_)
      execute(this->f_broadcast_a_, {inputs[0]}, {&a_broadcast});
    if (this->f_broadcast_b_)
      execute(this->f_broadcast_b_, {inputs[1]}, {&b_broadcast});
    // BMM backward
    const Tc *b =
        this->f_broadcast_b_ ? _get_data(&b_broadcast) : _get_data(inputs[1]);
    Tc *da = this->f_broadcast_a_ ? _cast_grad(&a_broadcast, true)
                                  : _cast_grad(inputs[0], !accum[0]);
    cuda_gemm_strided_batched<Tc>(
        this->device_, da, !this->transpose_a_, dy, this->col_y_, this->row_y_,
        true, b, this->col_b_, this->row_b_, this->transpose_b_, 1,
        ((accum[0] && !this->f_broadcast_a_) ? 1 : 0), this->samples_);
    // Broadcast backward
    if (this->f_broadcast_a_)
      nbla::backward(this->f_broadcast_a_, {inputs[0]}, {&a_broadcast}, {true},
                     {accum[0]});
  }
  if (propagate_down[1]) {
    // Broadcast
    Variable a_broadcast;
    Variable b_broadcast;
    if (this->f_broadcast_a_)
      execute(this->f_broadcast_a_, {inputs[0]}, {&a_broadcast});
    if (this->f_broadcast_b_)
      execute(this->f_broadcast_b_, {inputs[1]}, {&b_broadcast});
    // BMM backward
    const Tc *a =
        this->f_broadcast_a_ ? _get_data(&a_broadcast) : _get_data(inputs[0]);
    Tc *db = this->f_broadcast_b_ ? _cast_grad(&b_broadcast, true)
                                  : _cast_grad(inputs[1], !accum[1]);
    cuda_gemm_strided_batched<Tc>(
        this->device_, db, !this->transpose_b_, a, this->col_a_, this->row_a_,
        this->transpose_a_, dy, this->col_y_, this->row_y_, true, 1,
        ((accum[1] && !this->f_broadcast_b_) ? 1 : 0), this->samples_);
    // Broadcast backward
    if (this->f_broadcast_b_)
      nbla::backward(this->f_broadcast_b_, {inputs[1]}, {&b_broadcast}, {true},
                     {accum[1]});
  }
}

// ----------------------------------------------------------------------
// With cublas<t>gemmBatched
// ----------------------------------------------------------------------
#else // !(CUDA_VERSION >= 8000)
// Sets head pointers of matrices in mini-batch.
template <typename T>
__global__ void kernel_set_batch_pointers(int size, int stride, const T **ptr,
                                          const T *head) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { ptr[idx] = head + idx * stride; }
}

// A macro that creates an array of pointers of matrices.
#define NBLA_GET_BATCH_POINTERS(PTR, NAME, CONST)                              \
  NdArray list_##PTR(                                                          \
      Shape_t{static_cast<Size_t>(sizeof(Tc *) * this->samples_)});            \
  CONST Tc **dev_list_##PTR = reinterpret_cast<CONST Tc **>(                   \
      list_##PTR.cast(dtypes::BYTE, this->ctx_, true)->pointer<void>());       \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, this->samples_,    \
                                 this->offset_##NAME##_,                       \
                                 (const T **)dev_list_##PTR, (const T *)PTR)

template <typename T>
void BatchMatmulCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tc *a = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *b = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  NBLA_GET_BATCH_POINTERS(a, a, const); // dev_list_a
  NBLA_GET_BATCH_POINTERS(b, b, const); // dev_list_b
  NBLA_GET_BATCH_POINTERS(y, y, );      // dev_list_y
  // Apply GEMM to batch samples
  // Y{i} = op(A{i}) op(B{i}) where op is a transpose operator.
  cuda_gemm_batched<Tc>(this->device_, dev_list_y, true, dev_list_a,
                        this->col_a_, this->row_a_, !this->transpose_a_,
                        dev_list_b, this->col_b_, this->row_b_,
                        !this->transpose_b_, 1, 0, this->samples_);
}

template <typename T>
void BatchMatmulCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  NBLA_GET_BATCH_POINTERS(dy, y, const); // dev_list_dy
  if (propagate_down[0]) {
    const Tc *b = inputs[1]->get_data_pointer<Tc>(this->ctx_);
    Tc *da = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
    NBLA_GET_BATCH_POINTERS(b, b, const); // dev_list_b
    NBLA_GET_BATCH_POINTERS(da, a, );     // dev_list_da
    // op(dA{i}) (+)= dY{i} op(B{i}^T)
    cuda_gemm_batched<Tc>(this->device_, dev_list_da, !this->transpose_a_,
                          dev_list_dy, this->col_y_, this->row_y_, true,
                          dev_list_b, this->col_b_, this->row_b_,
                          this->transpose_b_, 1, (accum[0] ? 1 : 0),
                          this->samples_);
  }
  if (propagate_down[1]) {
    const Tc *a = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    Tc *db = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[1]);
    NBLA_GET_BATCH_POINTERS(a, a, const); // dev_list_a
    NBLA_GET_BATCH_POINTERS(db, b, );     // dev_list_db
    // op(dB{i}) (+)= dY{i} op(A{i}^T)
    cuda_gemm_batched<Tc>(this->device_, dev_list_db, !this->transpose_b_,
                          dev_list_a, this->col_a_, this->row_a_,
                          this->transpose_a_, dev_list_dy, this->col_y_,
                          this->row_y_, true, 1, (accum[1] ? 1 : 0),
                          this->samples_);
  }
}
#endif // CUDA_VERSION >= 8000
}
