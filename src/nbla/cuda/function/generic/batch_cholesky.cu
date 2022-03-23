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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/batch_cholesky.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

// Sets head pointers of matrices in mini-batch.
template <typename T>
__global__ void kernel_set_batch_pointers(int batchSize, int n, const T **ptr,
                                          const T *head)
{
  NBLA_CUDA_KERNEL_LOOP(idx, batchSize) { ptr[idx] = head + idx * n * n; }
}

// A macro that creates an array of pointers of matrices.
#define NBLA_GET_BATCH_POINTERS(PTR, NAME, BATCH, CONST)                       \
  NdArray list_##PTR(Shape_t{static_cast<Size_t>(sizeof(Tcu *) * BATCH)});     \
  CONST Tcu **dev_list_##NAME = reinterpret_cast<CONST Tcu **>(                \
      list_##PTR.cast(dtypes::BYTE, this->ctx_, true)->pointer<void>());       \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, BATCH, this->dim_, \
                                 (const T **)dev_list_##NAME, (const T *)PTR)

namespace nbla
{

  namespace batch_cholesky
  {
    template <typename T, bool upper>
    __global__ void construct_triangular_matrix(int batchSize, int n, const T *lu,
                                                T *y)
    {
      NBLA_CUDA_KERNEL_LOOP(idx, batchSize * n * n)
      {
        int matrixSize = (n * n);
        int batchNum = idx / matrixSize;
        int idxInMatrix = idx - batchNum * matrixSize;
        int row = idxInMatrix / n;
        int col = idxInMatrix % n;

        if (row < col)
        {
          // upper part
          y[idx] = upper ? lu[idx] : (T)0.0;
        }
        else if (col < row)
        {
          // lower part
          y[idx] = upper ? (T)0.0 : lu[batchNum * matrixSize + col * n + row];
        }
        else
        {
          // diagonal part
          y[idx] = lu[idx];
        }
      }
    }

    template <typename T>
    __global__ void apply_phi(int batchSize, int n, T *y)
    {
      NBLA_CUDA_KERNEL_LOOP(idx, batchSize * n * n)
      {
        int matrixSize = (n * n);
        int batchNum = idx / matrixSize;
        int idxInMatrix = idx - batchNum * matrixSize;
        int row = idxInMatrix / n;
        int col = idxInMatrix % n;

        if (row == col)
        {
          y[idx] = y[idx] * 0.5;
        }
        else if (row < col)
        {
          y[idx] = 0.0;
        }
      }
    }
  }

  template <typename T>
  void BatchCholeskyCuda<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs)
  {
    BatchCholesky<T>::setup_impl(inputs, outputs);

    cuda_set_device(this->device_);
  }

  template <typename T>
  void BatchCholeskyCuda<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs)
  {
    cuda_set_device(this->device_);

    NdArray info_ndarr(Shape_t{this->batch_size_});
    shared_ptr<Array> info =
        info_ndarr.cast_sp(dtypes::INT, this->ctx_, true /*write only*/);
    info->zero();

    NdArray lu(Shape_t{inputs[0]->size()});
    ArrayPtr lu_arr = lu.cast_sp(get_dtype<Tcu>(), this->ctx_, true);
    lu_arr->copy_from(
        inputs[0]->data()->cast(get_dtype<Tcu>(), this->ctx_, false));
    Tcu *lu_ptr = lu_arr->pointer<Tcu>();

    NBLA_GET_BATCH_POINTERS(lu_ptr, lu, this->batch_size_, ); // dev_list_lu

    // cholesky decomposition
    // NOTE: cusolver always return upper triangular part of the matrix
    cuda_potrf_batched<Tcu>(this->device_, this->dim_, dev_list_lu,
                            info->pointer<int>(), this->batch_size_);
    Tcu *y_ptr = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    if (this->upper_)
    {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (batch_cholesky::construct_triangular_matrix<Tcu, true>),
          this->batch_size_, this->dim_, lu_ptr, y_ptr);
    }
    else
    {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (batch_cholesky::construct_triangular_matrix<Tcu, false>),
          this->batch_size_, this->dim_, lu_ptr, y_ptr);
    }
  }

  template <typename T>
  void BatchCholeskyCuda<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum)
  {
    cuda_set_device(this->device_);
    BatchCholesky<T>::backward_impl(inputs, outputs, propagate_down, accum);
  }

  template <typename T>
  void BatchCholeskyCuda<T>::phi(Variable &x)
  {
    cuda_set_device(this->device_);
    Tcu *y = x.cast_data_and_get_pointer<Tcu>(this->ctx_, true);

    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((batch_cholesky::apply_phi<Tcu>),
                                   this->batch_size_, this->dim_, y);
  }
}
