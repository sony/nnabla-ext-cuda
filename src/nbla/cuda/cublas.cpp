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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cublas.hpp>

#include <cublas_v2.h>

namespace nbla {

template <>
void cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t op_x,
                        cublasOperation_t op_y, int m, int n, int k,
                        const float *alpha, const float *x, int lda,
                        const float *y, int ldb, const float *beta, float *z,
                        int ldc) {
  NBLA_CUBLAS_CHECK(cublasSgemm(handle, op_x, op_y, m, n, k, alpha, x, lda, y,
                                ldb, beta, z, ldc));
}

template <>
void cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t op_x,
                         cublasOperation_t op_y, int m, int n, int k,
                         const double *alpha, const double *x, int lda,
                         const double *y, int ldb, const double *beta,
                         double *z, int ldc) {
  NBLA_CUBLAS_CHECK(cublasDgemm(handle, op_x, op_y, m, n, k, alpha, x, lda, y,
                                ldb, beta, z, ldc));
}

template <>
void cublas_gemv<float>(cublasHandle_t handle, cublasOperation_t trans, int m,
                        int n, const float *alpha, const float *A, int lda,
                        const float *x, int incx, const float *beta, float *y,
                        int incy) {
  NBLA_CUBLAS_CHECK(
      cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

template <>
void cublas_gemv<double>(cublasHandle_t handle, cublasOperation_t trans, int m,
                         int n, const double *alpha, const double *A, int lda,
                         const double *x, int incx, const double *beta,
                         double *y, int incy) {
  NBLA_CUBLAS_CHECK(
      cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

template <>
void cublas_dot<float>(cublasHandle_t handle, int n, const float *x, int incx,
                       const float *y, int incy, float *out) {
  NBLA_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, out));
}

template <>
void cublas_dot<double>(cublasHandle_t handle, int n, const double *x, int incx,
                        const double *y, int incy, double *out) {
  NBLA_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, out));
}

template <>
void cublas_gemm_batched<float>(cublasHandle_t handle, cublasOperation_t op_x,
                                cublasOperation_t op_y, int m, int n, int k,
                                const float *alpha, const float **x, int lda,
                                const float **y, int ldb, const float *beta,
                                float **z, int ldc, int batch_count) {
  NBLA_CUBLAS_CHECK(cublasSgemmBatched(handle, op_x, op_y, m, n, k, alpha, x,
                                       lda, y, ldb, beta, z, ldc, batch_count));
}

template <>
void cublas_gemm_batched<double>(cublasHandle_t handle, cublasOperation_t op_x,
                                 cublasOperation_t op_y, int m, int n, int k,
                                 const double *alpha, const double **x, int lda,
                                 const double **y, int ldb, const double *beta,
                                 double **z, int ldc, int batch_count) {
  NBLA_CUBLAS_CHECK(cublasDgemmBatched(handle, op_x, op_y, m, n, k, alpha, x,
                                       lda, y, ldb, beta, z, ldc, batch_count));
}

#if CUDA_VERSION >= 8000
template <>
void cublas_gemm_strided_batched<float>(
    cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,
    int m, int n, int k, const float *alpha, const float *x, int lda,
    int stride_a, const float *y, int ldb, int stride_b, const float *beta,
    float *z, int ldc, int stride_c, int batch_count) {
  NBLA_CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, op_x, op_y, m, n, k, alpha, x, lda, stride_a, y, ldb, stride_b,
      beta, z, ldc, stride_c, batch_count));
}

template <>
void cublas_gemm_strided_batched<double>(
    cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,
    int m, int n, int k, const double *alpha, const double *x, int lda,
    int stride_a, const double *y, int ldb, int stride_b, const double *beta,
    double *z, int ldc, int stride_c, int batch_count) {
  NBLA_CUBLAS_CHECK(cublasDgemmStridedBatched(
      handle, op_x, op_y, m, n, k, alpha, x, lda, stride_a, y, ldb, stride_b,
      beta, z, ldc, stride_c, batch_count));
}
#endif
}
