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

#include <nbla/common.hpp>
#include <nbla/cuda/cublas.hpp>
#include <nbla/cuda/cuda.hpp>

namespace nbla {

#define _TD() typedef typename CudaNativeType<T>::type Tc
#define _RC(X) reinterpret_cast<Tc *>(X)
#define _RCC(X) reinterpret_cast<const Tc *>(X)

/**
*/
template <typename T>
void cuda_gemm(int device, T *z, bool transpose_z, const T *x, int row_x,
               int col_x, bool transpose_x, const T *y, int row_y, int col_y,
               bool transpose_y, float alpha, float beta) {
  if (transpose_z) {
    cuda_gemm<T>(device, z, false, y, row_y, col_y, !transpose_y, x, row_x,
                 col_x, !transpose_x, alpha, beta);
    return;
  }
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_y = transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transpose_x ? col_x : row_x;
  int n = transpose_y ? row_y : col_y;
  int k = transpose_x ? row_x : col_x;
  int l = transpose_y ? col_y : row_y;
  NBLA_CHECK(l == k, error_code::unclassified, "");
  cublas_gemm<Tc>(handle, op_x, op_y, m, n, k, alpha, _RCC(x), row_x, _RCC(y),
                  row_y, beta, _RC(z), m);
}

/**
*/
template <typename T>
void cuda_gemv(int device, T *z, const T *x, int row_x, int col_x,
               bool transpose_x, const T *y, int row_y, float alpha, float beta,
               int incy = 1, int incz = 1) {
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = row_x;
  int n = col_x;
  int k = transpose_x ? row_x : col_x;
  NBLA_CHECK(k == row_y, error_code::unclassified, "");
  cublas_gemv<Tc>(handle, op_x, m, n, alpha, _RCC(x), row_x, _RCC(y), incy,
                  beta, _RC(z), incz);
}

// cuBLAS does not have cublasHgemv. Use cublasHgemm instead.
// TODO: Check availability in new releases.
template <>
inline void cuda_gemv<HalfCuda>(int device, HalfCuda *z, const HalfCuda *x,
                                int row_x, int col_x, bool transpose_x,
                                const HalfCuda *y, int row_y, float alpha,
                                float beta, int incy, int incz) {
  cuda_gemm<HalfCuda>(device, z, false, x, row_x, col_x, transpose_x, y, row_y,
                      1, false, alpha, beta);
}

/**
 */
template <typename T>
void cuda_dot(int device, T *z, const T *x, int n, const T *y, int incx = 1,
              int incy = 1) {
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublas_dot<Tc>(handle, n, _RCC(x), incx, _RCC(y), incy, _RC(z));
}

/**
*/
template <typename T>
void cuda_gemm_batched(int device, T **z, bool transpose_z, const T **x,
                       int row_x, int col_x, bool transpose_x, const T **y,
                       int row_y, int col_y, bool transpose_y, float alpha,
                       float beta, int batch_count) {
  if (transpose_z) {
    cuda_gemm_batched<T>(device, z, false, y, row_y, col_y, !transpose_y, x,
                         row_x, col_x, !transpose_x, alpha, beta, batch_count);
    return;
  }
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_y = transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transpose_x ? col_x : row_x;
  int n = transpose_y ? row_y : col_y;
  int k = transpose_x ? row_x : col_x;
  int l = transpose_y ? col_y : row_y;
  NBLA_CHECK(l == k, error_code::unclassified, "");
  cublas_gemm_batched<Tc>(handle, op_x, op_y, m, n, k, alpha,
                          reinterpret_cast<const Tc **>(x), row_x,
                          reinterpret_cast<const Tc **>(y), row_y, beta,
                          reinterpret_cast<Tc **>(z), m, batch_count);
}

#if CUDA_VERSION >= 8000
/**
*/
template <typename T>
void cuda_gemm_strided_batched(int device, T *z, bool transpose_z, const T *x,
                               int row_x, int col_x, bool transpose_x,
                               const T *y, int row_y, int col_y,
                               bool transpose_y, float alpha, float beta,
                               int batch_count) {
  if (transpose_z) {
    cuda_gemm_strided_batched<T>(device, z, false, y, row_y, col_y,
                                 !transpose_y, x, row_x, col_x, !transpose_x,
                                 alpha, beta, batch_count);
    return;
  }
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  cublasOperation_t op_x = transpose_x ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_y = transpose_y ? CUBLAS_OP_T : CUBLAS_OP_N;
  int m = transpose_x ? col_x : row_x;
  int n = transpose_y ? row_y : col_y;
  int k = transpose_x ? row_x : col_x;
  int l = transpose_y ? col_y : row_y;
  NBLA_CHECK(l == k, error_code::unclassified, "");
  cublas_gemm_strided_batched<Tc>(
      handle, op_x, op_y, m, n, k, alpha, _RCC(x), row_x, row_x * col_x,
      _RCC(y), row_y, row_y * col_y, beta, _RC(z), m, m * n, batch_count);
}
#endif

template <typename T>
void cuda_getrf_batched(int device, int n, T **x, int *pivot, int *info,
                        int batchSize) {
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  // optimizing lda leaves for future improvement
  cublas_getrf_batched<Tc>(handle, n, reinterpret_cast<Tc **>(x), n, pivot,
                           info, batchSize);
}

template <typename T>
void cuda_getri_batched(int device, int n, const T **x, int *pivot, T **y,
                        int *info, int batchSize) {
  _TD();
  cublasHandle_t handle = SingletonManager::get<Cuda>()->cublas_handle(device);
  // optimizing lda and ldc leaves for future improvement
  cublas_getri_batched<Tc>(handle, n, reinterpret_cast<const Tc **>(x), n,
                           pivot, reinterpret_cast<Tc **>(y), n, info,
                           batchSize);
}
}
#endif
