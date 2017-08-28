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

#ifndef __NBLA_CUDA_MATH_HPP__
#define __NBLA_CUDA_MATH_HPP__
#include <nbla/cuda/cublas.hpp>

namespace nbla {

/**
*/
template <typename T>
void cuda_gemm(int device, T *z, bool transpose_z, const T *x, int row_x,
               int col_x, bool transpose_x, const T *y, int row_y, int col_y,
               bool transpose_y, T alpha, T beta) {
  if (transpose_z) {
    cuda_gemm<T>(device, z, false, y, row_y, col_y, !transpose_y, x, row_x,
                 col_x, !transpose_x, alpha, beta);
    return;
  }
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_y = transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transpose_x ? col_x : row_x;
  int n = transpose_y ? row_y : col_y;
  int k = transpose_x ? row_x : col_x;
  int l = transpose_y ? col_y : row_y;
  NBLA_CHECK(l == k, error_code::unclassified, "");
  cublas_gemm<T>(handle, op_x, op_y, m, n, k, &alpha, x, row_x, y, row_y, &beta,
                 z, m);
}

/**
*/
template <typename T>
void cuda_gemv(int device, T *z, const T *x, int row_x, int col_x,
               bool transpose_x, const T *y, int row_y, T alpha, T beta,
               int incy = 1, int incz = 1) {
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = row_x;
  int n = col_x;
  int k = transpose_x ? row_x : col_x;
  NBLA_CHECK(k == row_y, error_code::unclassified, "");
  cublas_gemv<T>(handle, op_x, m, n, &alpha, x, row_x, y, incy, &beta, z, incz);
}

/**
 */
template <typename T>
void cuda_dot(int device, T *z, const T *x, int n, const T *y, int incx = 1,
              int incy = 1) {
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublas_dot<T>(handle, n, x, incx, y, incy, z);
}

/**
*/
template <typename T>
void cuda_gemm_batched(int device, T **z, bool transpose_z, const T **x,
                       int row_x, int col_x, bool transpose_x, const T **y,
                       int row_y, int col_y, bool transpose_y, T alpha, T beta,
                       int batch_count) {
  if (transpose_z) {
    cuda_gemm_batched<T>(device, z, false, y, row_y, col_y, !transpose_y, x,
                         row_x, col_x, !transpose_x, alpha, beta, batch_count);
    return;
  }
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_y = transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transpose_x ? col_x : row_x;
  int n = transpose_y ? row_y : col_y;
  int k = transpose_x ? row_x : col_x;
  int l = transpose_y ? col_y : row_y;
  NBLA_CHECK(l == k, error_code::unclassified, "");
  cublas_gemm_batched<T>(handle, op_x, op_y, m, n, k, &alpha, x, row_x, y,
                         row_y, &beta, z, m, batch_count);
}
}
#endif
