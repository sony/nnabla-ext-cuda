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
#include <nbla/cuda/function/batch_det.hpp>
#include <nbla/variable.hpp>
#include <nbla/cuda/math.hpp>


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
  CONST Tc **dev_list_##NAME =                                                 \
      reinterpret_cast<CONST Tc **>(list_##PTR.pointer<void>());               \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, BATCH, dim_,       \
                                 (const T **)dev_list_##NAME, (const T *)PTR)

template <typename T>
__global__ void kernel_compute_det(int batchSize, int n, T *y,
                                   const T *lu, int *pivot) {
  NBLA_CUDA_KERNEL_LOOP(idx, batchSize) {
    y[idx] = 1.0;
    int offset = idx * n * n;
    int parity = 0;
    // determinant is a product of diagonal entries of lu factorization
    for (int i = 0; i < n; ++i) {
      y[idx] *= lu[offset + i * n + i];
      if (pivot[idx * n + i] != (i + 1))
        ++parity;
    }
    // flip sign dependently on pivot array that is equal to its index
    T sign = 1.0 - 2.0 * (T) (parity % 2);
    y[idx] *= sign;
  }
}

template <typename T>
void BatchDetCuda<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  BatchDet<T>::setup_impl(inputs, outputs);
  batch_size_ = inputs[0]->shape()[0];
  dim_ = inputs[0]->shape()[1];
}

// ----------------------------------------------------------------------
// With cublas<t>getrfBatched
// ----------------------------------------------------------------------
template <typename T>
void BatchDetCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  shared_ptr<CudaCachedArray> pivot =
    make_shared<CudaCachedArray>(dim_ * batch_size_, dtypes::INT, this->ctx_);
  pivot->zero();

  shared_ptr<CudaCachedArray> info =
    make_shared<CudaCachedArray>(batch_size_, dtypes::INT, this->ctx_);
  info->zero();

  shared_ptr<CudaCachedArray> lu =
    make_shared<CudaCachedArray>(inputs[0]->size(), get_dtype<Tc>(),
                                 this->ctx_);
  lu->copy_from(inputs[0]->data()->cast(get_dtype<Tc>(), this->ctx_, false));

  Tc* lu_ptr = lu->pointer<Tc>();
  NBLA_GET_BATCH_POINTERS(lu_ptr, lu, batch_size_, );     // dev_list_lu

  // LU factorization
  cuda_getrf_batched<Tc>(this->device_, dim_, dev_list_lu,
                         pivot->pointer<int>(), info->pointer<int>(),
                         batch_size_);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_compute_det, batch_size_, dim_, y,
                                 lu_ptr, pivot->pointer<int>());
}


template <typename T>
void BatchDetCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  cuda_set_device(this->device_);
  BatchDet<T>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
