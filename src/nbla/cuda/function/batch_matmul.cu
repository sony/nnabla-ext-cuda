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
#include <nbla/variable.hpp>

namespace nbla {

// Sets head pointers of matrices in mini-batch.
template <typename T>
__global__ void kernel_set_batch_pointers(int size, int stride, T **ptr,
                                          T *head) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { ptr[idx] = head + idx * stride; }
}

// A macro that creates an array of pointers of matrices.
#define NBLA_GET_BATCH_POINTERS(PTR, NAME, CONST)                              \
  CudaCachedArray list_##PTR(sizeof(T *) * this->samples_, dtypes::BYTE,       \
                             this->ctx_);                                      \
  CONST T **dev_list_##PTR =                                                   \
      reinterpret_cast<CONST T **>(list_##PTR.pointer<void>());                \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, this->samples_,    \
                                 this->offset_##NAME##_, dev_list_##PTR, PTR)

template <typename T>
void BatchMatmulCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);
  const T *a = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *b = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  NBLA_GET_BATCH_POINTERS(a, a, const); // dev_list_a
  NBLA_GET_BATCH_POINTERS(b, b, const); // dev_list_b
  NBLA_GET_BATCH_POINTERS(y, y, );      // dev_list_y
  // Apply GEMM to batch samples
  // Y{i} = op(A{i}) op(B{i}) where op is a transpose operator.
  cuda_gemm_batched<T>(this->device_, dev_list_y, true, dev_list_a,
                       this->col_a_, this->row_a_, !this->transpose_a_,
                       dev_list_b, this->col_b_, this->row_b_,
                       !this->transpose_b_, (T)1, (T)0, this->samples_);
}

template <typename T>
void BatchMatmulCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  NBLA_GET_BATCH_POINTERS(dy, y, const); // dev_list_dy
  if (propagate_down[0]) {
    const T *b = inputs[1]->get_data_pointer<T>(this->ctx_);
    T *da = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    NBLA_GET_BATCH_POINTERS(b, b, const); // dev_list_b
    NBLA_GET_BATCH_POINTERS(da, a, );     // dev_list_da
    const T beta = accum[0] ? 1 : 0;
    // op(dA{i}) (+)= dY{i} op(B{i}^T)
    cuda_gemm_batched<T>(this->device_, dev_list_da, !this->transpose_a_,
                         dev_list_dy, this->col_y_, this->row_y_, true,
                         dev_list_b, this->col_b_, this->row_b_,
                         this->transpose_b_, (T)1, beta, this->samples_);
  }
  if (propagate_down[1]) {
    const T *a = inputs[0]->get_data_pointer<T>(this->ctx_);
    T *db = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_);
    NBLA_GET_BATCH_POINTERS(a, a, const); // dev_list_a
    NBLA_GET_BATCH_POINTERS(db, b, );     // dev_list_db
    const T beta = accum[1] ? 1 : 0;
    // op(dB{i}) (+)= dY{i} op(A{i}^T)
    cuda_gemm_batched<T>(this->device_, dev_list_db, !this->transpose_b_,
                         dev_list_a, this->col_a_, this->row_a_,
                         this->transpose_a_, dev_list_dy, this->col_y_,
                         this->row_y_, true, (T)1, beta, this->samples_);
  }
}

// template instantiation
template class BatchMatmulCuda<float>;
}
