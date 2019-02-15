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
#include <nbla/cuda/function/inverse.hpp>
#include <nbla/cuda/math.hpp>
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
  CudaCachedArray list_##PTR(sizeof(Tc *) * BATCH, dtypes::BYTE,               \
                             this->ctx_);                                      \
  CONST Tc **dev_list_##PTR =                                                  \
      reinterpret_cast<CONST Tc **>(list_##PTR.pointer<void>());               \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, BATCH,             \
                                 _dim,                                         \
                                 (const T **)dev_list_##PTR, (const T *)PTR)   \

// ----------------------------------------------------------------------
// With cublas<t>getrfBatched and cublas<t>getriBatched
// ----------------------------------------------------------------------
template <typename T>
void InverseCuda<T>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
  Inverse<T>::setup_imple(inputs, outputs);
  // for backward
  inv_x_ = make_shared<Variable>();
  matmul1_out_ = make_shared<Variable>();
  f_batch_matmul1_ = create_BatchMatmul(this->ctx_, true, false);
  f_batch_matmul1_->setup(Variables{inv_x_, make_shared<Variable>(outputs[0]->grad())}, Variables{matmul1_out});
  f_batch_matmul2_ = create_BatchMatmul(this->ctx_, false, true);
  f_batch_matmul2_->setup(Variables{matmul1_out_, inv_x_}, Variables{make_shared<Variable>(inputs[0]->grad())});
}

template <typename T>
void InverseCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  int batchSize = inputs[0]->shape()[0];
  CudaCachedArray pivot(sizeof(int) * batchSize, dtypes::BYTE, this->ctx);
  CudaCachedArray info(sizeof(int) * dim_ * batchSize, dtypes::BYTE,
                       this->ctx);

  NBLA_GET_BATCH_POINTERS(x, x, batchSize, const);      // dev_list_x
  NBLA_GET_BATCH_POINTERS(y, y, batchSize, );           // dev_list_y

  // LU factorization
  cuda_getrf_batched<Tc>(this->device_, dim_, dev_list_x, pivot.pointer<int>(),
                         info.pointer<int>(), batchSize);
  // matrix inversion
  cuda_getri_batched<Tc>(this->device_, dim_, dev_list_x, pivot.pointer<int>(),
                         dev_list_y, info.pointer<int>(), batchSize);
}

template <typename T>
void InverseCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T* dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T* x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T* dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  int batchSize = inputs[0]->shape()[0];
  CudaCachedArray pivot(sizeof(int) * batchSize, dtypes::BYTE, this->ctx);
  CudaCachedArray info(sizeof(int) * dim_ * batchSize, dtypes::BYTE,
                       this->ctx);
  CudaCachedArray inv_x(sizeof(Tc) * dim_ * dim_ * batchSize, dtypes::BYTE,
                        this->ctx);
  CudaCachedArray inv_x(sizeof(Tc) * dim_ * dim_ * batchSize, dtypes::BYTE,

  NBLA_GET_BATCH_POINTERS(x, x, batchSize, const);      // dev_list_x
  NBLA_GET_BATCH_POINTERS(inv_x.get().pointer<Tc>(), batchSize, inv_x, );   // dev_list_inv_x

  // LU factorization
  cuda_getrf_batched<Tc>(this->device_, dim_, dev_list_x, pivot, info,
                         batchSize);
  // matrix inversion
  cuda_getri_batched<Tc>(this->device_, dim_, dev_list_x, pivot, dev_list_inv_x,
                         info, batchSize);

  f_matmul1_->forward(Variables{inv_x_, make_shared<Variable>(outputs[0]->grad())}, Variables{matmul1_out});
  f_matmul2_->forward(Variables{matmul1_out_, inv_x_}, Variables{make_shared<Variable>(inputs[0]->grad())});
}
}
