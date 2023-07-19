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

#include <iostream>
#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/batch_inv.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/function/batch_matmul.hpp>
#include <nbla/variable.hpp>

namespace nbla {

// Sets head pointers of matrices in mini-batch.
template <typename T>
__global__ void kernel_set_batch_pointers(int batchSize, int n, const T **ptr,
                                          const T *head) {
  NBLA_CUDA_KERNEL_LOOP(idx, batchSize) { ptr[idx] = head + idx * n * n; }
}

// A macro that creates an array of pointers of matrices.
#define NBLA_GET_BATCH_POINTERS(PTR, NAME, BATCH, CONST)                       \
  NdArray list_##PTR(Shape_t{static_cast<Size_t>(sizeof(Tc *) * BATCH)});      \
  CONST Tc **dev_list_##NAME = reinterpret_cast<CONST Tc **>(                  \
      list_##PTR.cast(dtypes::BYTE, this->ctx_, true)->pointer<void>());       \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, BATCH, dim_,       \
                                 (const T **)dev_list_##NAME, (const T *)PTR)

// ----------------------------------------------------------------------
// With cublas<t>getrfBatched and cublas<t>getriBatched
// ----------------------------------------------------------------------
template <typename T>
void BatchInvCuda<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  BatchInv<T>::setup_impl(inputs, outputs);
  batch_size_ = inputs[0]->shape()[0];
  dim_ = inputs[0]->shape()[1];
}

template <typename T>
void BatchInvCuda<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  NdArray pivot(Shape_t{dim_ * batch_size_});
  NdArray info(Shape_t{batch_size_});
  NdArray lu(Shape_t{inputs[0]->size()});

  int *pivot_ptr = pivot.cast(dtypes::INT, this->ctx_, true)->pointer<int>();
  int *info_ptr = info.cast(dtypes::INT, this->ctx_, true)->pointer<int>();
  ArrayPtr lu_arr = lu.cast_sp(get_dtype<Tc>(), this->ctx_, true);

  lu_arr->copy_from(
      inputs[0]->data()->cast(get_dtype<Tc>(), this->ctx_, false));

  Tc *lu_ptr = lu_arr->pointer<Tc>();
  NBLA_GET_BATCH_POINTERS(lu_ptr, lu, batch_size_, ) // dev_list_lu
  NBLA_GET_BATCH_POINTERS(y, y, batch_size_, );      // dev_list_y

  // LU factorization
  cuda_getrf_batched<Tc>(this->device_, dim_, dev_list_lu, pivot_ptr, info_ptr,
                         batch_size_);

  // matrix inversion
  cuda_getri_batched<Tc>(this->device_, dim_, (const Tc **)dev_list_lu,
                         pivot_ptr, dev_list_y, info_ptr, batch_size_);
}
} // namespace nbla
